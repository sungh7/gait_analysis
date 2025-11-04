import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/services.dart';
import 'package:google_ml_kit/google_ml_kit.dart';

import '../../core/constants/app_constants.dart';
import '../models/gait_analysis_models.dart';

/// MediaPipe 서비스 추상 클래스
abstract class MediaPipeService {
  Future<void> initialize();
  Future<List<PoseLandmark>> detectPose(Uint8List imageBytes);
  Future<List<PoseLandmark>> detectPoseFromImage(ui.Image image);
  Stream<List<PoseLandmark>> detectPoseStream(Stream<Uint8List> imageStream);
  Future<void> dispose();
  bool get isInitialized;
}

/// Google ML Kit을 사용한 MediaPipe 서비스 구현
class MediaPipeServiceImpl implements MediaPipeService {
  static const MethodChannel _channel = MethodChannel('gait_analysis/mediapipe');

  PoseDetector? _poseDetector;
  bool _isInitialized = false;
  StreamController<List<PoseLandmark>>? _poseStreamController;

  @override
  bool get isInitialized => _isInitialized;

  @override
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // Google ML Kit PoseDetector 초기화
      _poseDetector = PoseDetector(
        options: PoseDetectorOptions(
          model: PoseDetectionModel.accurate, // 높은 정확도 모델
          mode: PoseDetectionMode.stream,     // 스트리밍 모드
        ),
      );

      // 네이티브 MediaPipe 초기화
      await _initializeNativeMediaPipe();

      _isInitialized = true;
      print('✅ MediaPipe service initialized successfully');
    } catch (e) {
      throw MediaPipeServiceException('Failed to initialize MediaPipe: $e');
    }
  }

  Future<void> _initializeNativeMediaPipe() async {
    try {
      final result = await _channel.invokeMethod('initialize', {
        'modelPath': AppConstants.poseEstimationModelPath,
        'confidence': AppConstants.minConfidenceThreshold,
        'enableGpu': true,
      });

      if (result['success'] != true) {
        throw MediaPipeServiceException(result['error'] ?? 'Unknown initialization error');
      }
    } on PlatformException catch (e) {
      throw MediaPipeServiceException('Platform error: ${e.message}');
    }
  }

  @override
  Future<List<PoseLandmark>> detectPose(Uint8List imageBytes) async {
    if (!_isInitialized) {
      throw MediaPipeServiceException('MediaPipe not initialized');
    }

    try {
      // Google ML Kit을 사용한 포즈 검출
      final inputImage = InputImage.fromBytes(
        bytes: imageBytes,
        metadata: InputImageMetadata(
          size: const Size(224, 224),
          rotation: InputImageRotation.rotation0deg,
          format: InputImageFormat.yuv420,
          bytesPerRow: 224,
        ),
      );

      final poses = await _poseDetector!.processImage(inputImage);

      if (poses.isEmpty) {
        return [];
      }

      // 가장 확실한 포즈 선택
      final pose = poses.first;
      return _convertPoseLandmarks(pose.landmarks);
    } catch (e) {
      // Fallback to native MediaPipe
      return await _detectPoseNative(imageBytes);
    }
  }

  Future<List<PoseLandmark>> _detectPoseNative(Uint8List imageBytes) async {
    try {
      final result = await _channel.invokeMethod('detectPose', {
        'imageBytes': imageBytes,
        'width': 224,
        'height': 224,
      });

      if (result['success'] == true) {
        final landmarks = result['landmarks'] as List;
        return landmarks.map((landmark) => PoseLandmark(
          id: landmark['id'],
          x: landmark['x'].toDouble(),
          y: landmark['y'].toDouble(),
          z: landmark['z'].toDouble(),
          visibility: landmark['visibility'].toDouble(),
        )).toList();
      } else {
        throw MediaPipeServiceException(result['error'] ?? 'Native detection failed');
      }
    } on PlatformException catch (e) {
      throw MediaPipeServiceException('Native detection error: ${e.message}');
    }
  }

  @override
  Future<List<PoseLandmark>> detectPoseFromImage(ui.Image image) async {
    if (!_isInitialized) {
      throw MediaPipeServiceException('MediaPipe not initialized');
    }

    try {
      // UI.Image를 바이트로 변환
      final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
      if (byteData == null) {
        throw MediaPipeServiceException('Failed to convert image to bytes');
      }

      return await detectPose(byteData.buffer.asUint8List());
    } catch (e) {
      throw MediaPipeServiceException('Failed to detect pose from image: $e');
    }
  }

  @override
  Stream<List<PoseLandmark>> detectPoseStream(Stream<Uint8List> imageStream) {
    if (!_isInitialized) {
      throw MediaPipeServiceException('MediaPipe not initialized');
    }

    _poseStreamController?.close();
    _poseStreamController = StreamController<List<PoseLandmark>>.broadcast();

    imageStream.listen(
      (imageBytes) async {
        try {
          final landmarks = await detectPose(imageBytes);
          _poseStreamController?.add(landmarks);
        } catch (e) {
          _poseStreamController?.addError(e);
        }
      },
      onError: (error) {
        _poseStreamController?.addError(error);
      },
      onDone: () {
        _poseStreamController?.close();
      },
    );

    return _poseStreamController!.stream;
  }

  // Google ML Kit PoseLandmark을 우리 모델로 변환
  List<PoseLandmark> _convertPoseLandmarks(Map<PoseLandmarkType, PoseLandmark?> mlKitLandmarks) {
    final landmarks = <PoseLandmark>[];

    // MediaPipe 33개 랜드마크 순서대로 변환
    final landmarkOrder = [
      PoseLandmarkType.nose,
      PoseLandmarkType.leftEyeInner,
      PoseLandmarkType.leftEye,
      PoseLandmarkType.leftEyeOuter,
      PoseLandmarkType.rightEyeInner,
      PoseLandmarkType.rightEye,
      PoseLandmarkType.rightEyeOuter,
      PoseLandmarkType.leftEar,
      PoseLandmarkType.rightEar,
      PoseLandmarkType.leftMouth,
      PoseLandmarkType.rightMouth,
      PoseLandmarkType.leftShoulder,
      PoseLandmarkType.rightShoulder,
      PoseLandmarkType.leftElbow,
      PoseLandmarkType.rightElbow,
      PoseLandmarkType.leftWrist,
      PoseLandmarkType.rightWrist,
      PoseLandmarkType.leftPinky,
      PoseLandmarkType.rightPinky,
      PoseLandmarkType.leftIndex,
      PoseLandmarkType.rightIndex,
      PoseLandmarkType.leftThumb,
      PoseLandmarkType.rightThumb,
      PoseLandmarkType.leftHip,
      PoseLandmarkType.rightHip,
      PoseLandmarkType.leftKnee,
      PoseLandmarkType.rightKnee,
      PoseLandmarkType.leftAnkle,
      PoseLandmarkType.rightAnkle,
      PoseLandmarkType.leftHeel,
      PoseLandmarkType.rightHeel,
      PoseLandmarkType.leftFootIndex,
      PoseLandmarkType.rightFootIndex,
    ];

    for (int i = 0; i < landmarkOrder.length; i++) {
      final mlKitLandmark = mlKitLandmarks[landmarkOrder[i]];
      if (mlKitLandmark != null) {
        landmarks.add(PoseLandmark(
          id: i,
          x: mlKitLandmark.x,
          y: mlKitLandmark.y,
          z: mlKitLandmark.z ?? 0.0,
          visibility: mlKitLandmark.likelihood,
        ));
      } else {
        // 랜드마크가 검출되지 않은 경우 기본값
        landmarks.add(PoseLandmark(
          id: i,
          x: 0.0,
          y: 0.0,
          z: 0.0,
          visibility: 0.0,
        ));
      }
    }

    return landmarks;
  }

  // 보행 분석에 특화된 랜드마크 필터링
  List<PoseLandmark> filterGaitRelevantLandmarks(List<PoseLandmark> landmarks) {
    // 보행 분석에 중요한 랜드마크 인덱스들
    final gaitRelevantIndices = [
      11, 12, // 어깨
      23, 24, // 엉덩이
      25, 26, // 무릎
      27, 28, // 발목
      29, 30, // 발뒤꿈치
      31, 32, // 발가락
    ];

    return landmarks.where((landmark) =>
      gaitRelevantIndices.contains(landmark.id) &&
      landmark.visibility > AppConstants.minConfidenceThreshold
    ).toList();
  }

  // 랜드마크 평활화 (노이즈 제거)
  List<PoseLandmark> smoothLandmarks(
    List<PoseLandmark> currentLandmarks,
    List<List<PoseLandmark>> previousFrames,
    {double smoothingFactor = 0.7}
  ) {
    if (previousFrames.isEmpty) return currentLandmarks;

    final smoothedLandmarks = <PoseLandmark>[];

    for (int i = 0; i < currentLandmarks.length; i++) {
      final current = currentLandmarks[i];

      // 이전 프레임들의 평균 계산
      double avgX = 0, avgY = 0, avgZ = 0, avgVisibility = 0;
      int validFrames = 0;

      for (final frame in previousFrames) {
        if (i < frame.length) {
          avgX += frame[i].x;
          avgY += frame[i].y;
          avgZ += frame[i].z;
          avgVisibility += frame[i].visibility;
          validFrames++;
        }
      }

      if (validFrames > 0) {
        avgX /= validFrames;
        avgY /= validFrames;
        avgZ /= validFrames;
        avgVisibility /= validFrames;

        // 지수 이동 평균 적용
        smoothedLandmarks.add(PoseLandmark(
          id: current.id,
          x: current.x * (1 - smoothingFactor) + avgX * smoothingFactor,
          y: current.y * (1 - smoothingFactor) + avgY * smoothingFactor,
          z: current.z * (1 - smoothingFactor) + avgZ * smoothingFactor,
          visibility: current.visibility * (1 - smoothingFactor) + avgVisibility * smoothingFactor,
        ));
      } else {
        smoothedLandmarks.add(current);
      }
    }

    return smoothedLandmarks;
  }

  // 랜드마크 품질 평가
  double evaluateLandmarkQuality(List<PoseLandmark> landmarks) {
    if (landmarks.isEmpty) return 0.0;

    // 가시성 점수
    final visibilityScore = landmarks
        .map((l) => l.visibility)
        .reduce((a, b) => a + b) / landmarks.length;

    // 보행 분석 핵심 랜드마크 검출 여부
    final criticalLandmarks = [23, 24, 25, 26, 27, 28]; // 힙, 무릎, 발목
    final criticalDetected = criticalLandmarks
        .where((id) => landmarks
            .any((l) => l.id == id && l.visibility > AppConstants.minConfidenceThreshold))
        .length;

    final completenessScore = criticalDetected / criticalLandmarks.length;

    // 랜드마크 안정성 (연속 프레임 간 변화량)
    final stabilityScore = 0.8; // 실제로는 이전 프레임과 비교하여 계산

    return (visibilityScore * 0.4 + completenessScore * 0.4 + stabilityScore * 0.2)
        .clamp(0.0, 1.0);
  }

  // 실시간 성능 모니터링
  Future<MediaPipePerformanceMetrics> getPerformanceMetrics() async {
    try {
      final result = await _channel.invokeMethod('getPerformanceMetrics');

      return MediaPipePerformanceMetrics(
        averageInferenceTime: result['averageInferenceTime']?.toDouble() ?? 0.0,
        framesPerSecond: result['framesPerSecond']?.toDouble() ?? 0.0,
        memoryUsage: result['memoryUsage']?.toInt() ?? 0,
        cpuUsage: result['cpuUsage']?.toDouble() ?? 0.0,
        gpuUsage: result['gpuUsage']?.toDouble() ?? 0.0,
      );
    } catch (e) {
      return MediaPipePerformanceMetrics(
        averageInferenceTime: 0.0,
        framesPerSecond: 0.0,
        memoryUsage: 0,
        cpuUsage: 0.0,
        gpuUsage: 0.0,
      );
    }
  }

  @override
  Future<void> dispose() async {
    await _poseStreamController?.close();
    _poseStreamController = null;

    await _poseDetector?.close();
    _poseDetector = null;

    try {
      await _channel.invokeMethod('dispose');
    } catch (e) {
      print('Error disposing native MediaPipe: $e');
    }

    _isInitialized = false;
    print('🗑️ MediaPipe service disposed');
  }
}

/// MediaPipe 성능 메트릭
class MediaPipePerformanceMetrics {
  final double averageInferenceTime; // milliseconds
  final double framesPerSecond;
  final int memoryUsage; // bytes
  final double cpuUsage; // percentage (0-100)
  final double gpuUsage; // percentage (0-100)

  const MediaPipePerformanceMetrics({
    required this.averageInferenceTime,
    required this.framesPerSecond,
    required this.memoryUsage,
    required this.cpuUsage,
    required this.gpuUsage,
  });

  Map<String, dynamic> toJson() => {
    'averageInferenceTime': averageInferenceTime,
    'framesPerSecond': framesPerSecond,
    'memoryUsage': memoryUsage,
    'cpuUsage': cpuUsage,
    'gpuUsage': gpuUsage,
  };

  bool get isPerformant =>
      averageInferenceTime < 50 && // 50ms 이하
      framesPerSecond >= 25 &&     // 25fps 이상
      cpuUsage < 80;               // CPU 80% 이하
}

/// MediaPipe 서비스 예외 클래스
class MediaPipeServiceException implements Exception {
  final String message;
  MediaPipeServiceException(this.message);

  @override
  String toString() => 'MediaPipeServiceException: $message';
}