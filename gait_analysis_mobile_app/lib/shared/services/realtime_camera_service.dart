import 'dart:async';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import '../models/gait_analysis_models.dart';
import 'mediapipe_bridge.dart';
import 'v7_pure3d_service.dart';

/// 실시간 카메라 처리 서비스
///
/// 카메라 프레임을 실시간으로 캡처하고 MediaPipe로 처리
class RealtimeCameraService {
  final MediaPipeBridge _mediapipe = MediaPipeBridge();
  final V7Pure3DService _v7Service = V7Pure3DService();

  CameraController? _cameraController;
  bool _isProcessing = false;
  bool _isRecording = false;

  // 녹화된 랜드마크 시퀀스
  final List<List<PoseLandmark>> _recordedLandmarks = [];
  int _recordedFrames = 0;
  DateTime? _recordingStartTime;

  // 콜백
  Function(List<PoseLandmark>)? onLandmarksDetected;
  Function(String)? onStatusUpdate;
  Function(V7DetectionResult)? onAnalysisComplete;

  // 설정
  static const int targetFps = 30;
  static const int targetFrames = 180; // 6초 @ 30fps
  static const Duration frameDuration = Duration(milliseconds: 33); // ~30fps

  /// 초기화
  Future<void> initialize() async {
    try {
      // MediaPipe 초기화
      await _mediapipe.initialize();
      onStatusUpdate?.call('✅ MediaPipe 초기화 완료');

      // 카메라 초기화
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        throw CameraException('NO_CAMERA', 'No cameras available');
      }

      // 전면 카메라 선택
      final frontCamera = cameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      _cameraController = CameraController(
        frontCamera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await _cameraController!.initialize();
      onStatusUpdate?.call('✅ 카메라 초기화 완료');

    } catch (e) {
      onStatusUpdate?.call('❌ 초기화 실패: $e');
      rethrow;
    }
  }

  /// 녹화 시작
  Future<void> startRecording() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      throw StateError('Camera not initialized');
    }

    if (_isRecording) {
      throw StateError('Already recording');
    }

    _isRecording = true;
    _recordedLandmarks.clear();
    _recordedFrames = 0;
    _recordingStartTime = DateTime.now();

    onStatusUpdate?.call('🎥 촬영 중... (6초)');

    // 프레임 처리 시작
    await _cameraController!.startImageStream(_processImageFrame);

    // 6초 후 자동 정지
    Timer(const Duration(seconds: 6), () {
      if (_isRecording) {
        stopRecording();
      }
    });
  }

  /// 프레임 처리
  void _processImageFrame(CameraImage image) {
    if (_isProcessing || !_isRecording) return;

    if (_recordedFrames >= targetFrames) {
      stopRecording();
      return;
    }

    _isProcessing = true;

    _processCameraImage(image).then((landmarks) {
      if (landmarks != null && landmarks.isNotEmpty) {
        _recordedLandmarks.add(landmarks);
        _recordedFrames++;

        onLandmarksDetected?.call(landmarks);
        onStatusUpdate?.call('🎥 촬영 중... ($_recordedFrames/$targetFrames)');
      }

      _isProcessing = false;
    }).catchError((error) {
      print('❌ Frame processing error: $error');
      _isProcessing = false;
    });
  }

  /// 카메라 이미지를 MediaPipe로 처리
  Future<List<PoseLandmark>?> _processCameraImage(CameraImage image) async {
    try {
      // CameraImage를 JPEG로 변환
      final jpegBytes = await _convertCameraImageToJpeg(image);
      if (jpegBytes == null) return null;

      // MediaPipe로 Pose 검출
      final timestampMs = DateTime.now().millisecondsSinceEpoch;
      final landmarks = await _mediapipe.detectPoseVideo(
        jpegBytes,
        timestampMs: timestampMs,
      );

      return landmarks;
    } catch (e) {
      print('❌ Image processing error: $e');
      return null;
    }
  }

  /// CameraImage를 JPEG로 변환
  Future<Uint8List?> _convertCameraImageToJpeg(CameraImage image) async {
    try {
      // YUV420 to RGB
      final int width = image.width;
      final int height = image.height;

      // Create image
      final img.Image rgbImage = img.Image(width: width, height: height);

      // Convert YUV to RGB (simplified - production needs proper conversion)
      final Plane yPlane = image.planes[0];
      final Plane uPlane = image.planes[1];
      final Plane vPlane = image.planes[2];

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final int yIndex = y * yPlane.bytesPerRow + x;
          final int uvIndex = (y ~/ 2) * uPlane.bytesPerRow + (x ~/ 2);

          if (yIndex < yPlane.bytes.length &&
              uvIndex < uPlane.bytes.length &&
              uvIndex < vPlane.bytes.length) {
            final int yValue = yPlane.bytes[yIndex];
            final int uValue = uPlane.bytes[uvIndex];
            final int vValue = vPlane.bytes[uvIndex];

            // YUV to RGB conversion
            int r = (yValue + 1.370705 * (vValue - 128)).toInt();
            int g = (yValue - 0.337633 * (uValue - 128) - 0.698001 * (vValue - 128)).toInt();
            int b = (yValue + 1.732446 * (uValue - 128)).toInt();

            // Clamp values
            r = r.clamp(0, 255);
            g = g.clamp(0, 255);
            b = b.clamp(0, 255);

            rgbImage.setPixelRgba(x, y, r, g, b, 255);
          }
        }
      }

      // Encode to JPEG
      final jpegBytes = img.encodeJpg(rgbImage, quality: 85);
      return Uint8List.fromList(jpegBytes);

    } catch (e) {
      print('❌ Image conversion error: $e');
      return null;
    }
  }

  /// 녹화 중지
  Future<void> stopRecording() async {
    if (!_isRecording) return;

    _isRecording = false;

    try {
      await _cameraController?.stopImageStream();
      onStatusUpdate?.call('✅ 촬영 완료 - 분석 중...');

      // V7 Pure 3D 분석
      await _analyzeRecording();

    } catch (e) {
      print('❌ Stop recording error: $e');
      onStatusUpdate?.call('❌ 촬영 중지 실패: $e');
    }
  }

  /// 녹화된 데이터 분석
  Future<void> _analyzeRecording() async {
    if (_recordedLandmarks.isEmpty) {
      onStatusUpdate?.call('❌ 녹화된 데이터가 없습니다');
      return;
    }

    try {
      // V7 Pure 3D 특징 추출
      final features = _v7Service.extractFeatures(
        _recordedLandmarks,
        targetFps.toDouble(),
      );

      // 병리적 보행 검출
      final result = _v7Service.detectPathologicalGait(features);

      onAnalysisComplete?.call(result);
      onStatusUpdate?.call(result.isPathological
          ? '⚠️ 비정상 보행 패턴 검출'
          : '✅ 정상 보행 패턴');

    } catch (e) {
      print('❌ Analysis error: $e');
      onStatusUpdate?.call('❌ 분석 실패: $e');
    }
  }

  /// 정리
  Future<void> dispose() async {
    _isRecording = false;
    _isProcessing = false;

    try {
      if (_cameraController?.value.isStreamingImages ?? false) {
        await _cameraController?.stopImageStream();
      }
    } catch (e) {
      print('Warning: Failed to stop image stream: $e');
    }

    await _cameraController?.dispose();
    _cameraController = null;

    await _mediapipe.dispose();

    _recordedLandmarks.clear();
  }

  /// Getters
  CameraController? get cameraController => _cameraController;
  bool get isRecording => _isRecording;
  int get recordedFrames => _recordedFrames;
  List<List<PoseLandmark>> get recordedLandmarks => _recordedLandmarks;
}

/// 카메라 예외
class CameraException implements Exception {
  final String code;
  final String message;

  CameraException(this.code, this.message);

  @override
  String toString() => 'CameraException[$code]: $message';
}
