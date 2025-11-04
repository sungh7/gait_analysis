import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import '../../core/constants/app_constants.dart';
import '../models/gait_analysis_models.dart';

/// 카메라 서비스 추상 클래스
abstract class CameraService {
  Future<void> initialize();
  Future<void> startVideoRecording();
  Future<String?> stopVideoRecording();
  Future<void> dispose();

  Stream<Uint8List> get frameStream;
  bool get isRecording;
  bool get isInitialized;
  CameraController? get controller;
}

/// 카메라 서비스 구현 클래스
class CameraServiceImpl implements CameraService {
  CameraController? _controller;
  List<CameraDescription>? _cameras;
  StreamController<Uint8List>? _frameStreamController;
  Timer? _frameTimer;

  bool _isRecording = false;
  bool _isInitialized = false;
  bool _isStreaming = false;

  @override
  CameraController? get controller => _controller;

  @override
  bool get isRecording => _isRecording;

  @override
  bool get isInitialized => _isInitialized;

  @override
  Stream<Uint8List> get frameStream =>
      _frameStreamController?.stream ?? const Stream.empty();

  @override
  Future<void> initialize() async {
    try {
      // 권한 확인
      await _requestPermissions();

      // 카메라 목록 가져오기
      _cameras = await availableCameras();

      if (_cameras == null || _cameras!.isEmpty) {
        throw CameraServiceException('No cameras available');
      }

      // 후면 카메라 선택 (보행 분석에 적합)
      final CameraDescription camera = _cameras!.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.back,
        orElse: () => _cameras!.first,
      );

      // 카메라 컨트롤러 초기화
      _controller = CameraController(
        camera,
        ResolutionPreset.high,
        enableAudio: false, // 보행 분석에서는 오디오 불필요
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await _controller!.initialize();

      // 프레임 스트림 초기화
      _frameStreamController = StreamController<Uint8List>.broadcast();

      _isInitialized = true;
      print('✅ Camera service initialized successfully');
    } catch (e) {
      throw CameraServiceException('Failed to initialize camera: $e');
    }
  }

  Future<void> _requestPermissions() async {
    final permissions = [
      Permission.camera,
      Permission.microphone,
      Permission.storage,
    ];

    Map<Permission, PermissionStatus> statuses = await permissions.request();

    if (statuses[Permission.camera] != PermissionStatus.granted) {
      throw CameraServiceException('Camera permission denied');
    }
  }

  @override
  Future<void> startVideoRecording() async {
    if (!_isInitialized || _controller == null) {
      throw CameraServiceException('Camera not initialized');
    }

    if (_isRecording) {
      throw CameraServiceException('Already recording');
    }

    try {
      await _controller!.startVideoRecording();
      _isRecording = true;

      // 실시간 프레임 스트리밍 시작
      _startFrameStreaming();

      print('📹 Video recording started');
    } catch (e) {
      throw CameraServiceException('Failed to start recording: $e');
    }
  }

  void _startFrameStreaming() {
    if (_isStreaming) return;

    _isStreaming = true;

    // 30fps로 프레임 추출
    _frameTimer = Timer.periodic(
      const Duration(milliseconds: 33), // ~30fps
      (timer) async {
        if (!_isRecording || _controller == null) {
          _stopFrameStreaming();
          return;
        }

        try {
          // 현재 프레임 캡처
          final image = await _controller!.takePicture();
          final bytes = await image.readAsBytes();

          // 스트림에 프레임 추가
          _frameStreamController?.add(bytes);
        } catch (e) {
          print('Frame capture error: $e');
        }
      },
    );
  }

  void _stopFrameStreaming() {
    _frameTimer?.cancel();
    _frameTimer = null;
    _isStreaming = false;
  }

  @override
  Future<String?> stopVideoRecording() async {
    if (!_isRecording || _controller == null) {
      throw CameraServiceException('Not recording');
    }

    try {
      final file = await _controller!.stopVideoRecording();
      _isRecording = false;

      // 프레임 스트리밍 중지
      _stopFrameStreaming();

      print('🛑 Video recording stopped: ${file.path}');
      return file.path;
    } catch (e) {
      throw CameraServiceException('Failed to stop recording: $e');
    }
  }

  Future<void> pauseRecording() async {
    if (!_isRecording || _controller == null) {
      throw CameraServiceException('Not recording');
    }

    try {
      await _controller!.pauseVideoRecording();
      _stopFrameStreaming();
      print('⏸️ Video recording paused');
    } catch (e) {
      throw CameraServiceException('Failed to pause recording: $e');
    }
  }

  Future<void> resumeRecording() async {
    if (!_isRecording || _controller == null) {
      throw CameraServiceException('Not recording');
    }

    try {
      await _controller!.resumeVideoRecording();
      _startFrameStreaming();
      print('▶️ Video recording resumed');
    } catch (e) {
      throw CameraServiceException('Failed to resume recording: $e');
    }
  }

  Future<void> switchCamera() async {
    if (!_isInitialized || _cameras == null || _cameras!.length < 2) {
      throw CameraServiceException('Cannot switch camera');
    }

    try {
      final currentCamera = _controller!.description;
      final newCamera = _cameras!.firstWhere(
        (camera) => camera.lensDirection != currentCamera.lensDirection,
        orElse: () => currentCamera,
      );

      if (newCamera == currentCamera) {
        throw CameraServiceException('No alternative camera available');
      }

      // 기존 컨트롤러 해제
      await _controller!.dispose();

      // 새 카메라로 컨트롤러 재생성
      _controller = CameraController(
        newCamera,
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await _controller!.initialize();
      print('🔄 Camera switched successfully');
    } catch (e) {
      throw CameraServiceException('Failed to switch camera: $e');
    }
  }

  Future<void> setFlashMode(FlashMode mode) async {
    if (!_isInitialized || _controller == null) {
      throw CameraServiceException('Camera not initialized');
    }

    try {
      await _controller!.setFlashMode(mode);
    } catch (e) {
      throw CameraServiceException('Failed to set flash mode: $e');
    }
  }

  Future<void> setZoomLevel(double zoom) async {
    if (!_isInitialized || _controller == null) {
      throw CameraServiceException('Camera not initialized');
    }

    try {
      final maxZoom = await _controller!.getMaxZoomLevel();
      final minZoom = await _controller!.getMinZoomLevel();

      final clampedZoom = zoom.clamp(minZoom, maxZoom);
      await _controller!.setZoomLevel(clampedZoom);
    } catch (e) {
      throw CameraServiceException('Failed to set zoom level: $e');
    }
  }

  Future<double> getZoomLevel() async {
    if (!_isInitialized || _controller == null) {
      throw CameraServiceException('Camera not initialized');
    }

    try {
      return await _controller!.getZoomLevel();
    } catch (e) {
      throw CameraServiceException('Failed to get zoom level: $e');
    }
  }

  Future<void> setFocusMode(FocusMode mode) async {
    if (!_isInitialized || _controller == null) {
      throw CameraServiceException('Camera not initialized');
    }

    try {
      await _controller!.setFocusMode(mode);
    } catch (e) {
      throw CameraServiceException('Failed to set focus mode: $e');
    }
  }

  Future<void> setFocusPoint(Offset point) async {
    if (!_isInitialized || _controller == null) {
      throw CameraServiceException('Camera not initialized');
    }

    try {
      await _controller!.setFocusPoint(point);
    } catch (e) {
      throw CameraServiceException('Failed to set focus point: $e');
    }
  }

  Future<void> setExposureMode(ExposureMode mode) async {
    if (!_isInitialized || _controller == null) {
      throw CameraServiceException('Camera not initialized');
    }

    try {
      await _controller!.setExposureMode(mode);
    } catch (e) {
      throw CameraServiceException('Failed to set exposure mode: $e');
    }
  }

  Future<void> setExposureOffset(double offset) async {
    if (!_isInitialized || _controller == null) {
      throw CameraServiceException('Camera not initialized');
    }

    try {
      final maxOffset = await _controller!.getMaxExposureOffset();
      final minOffset = await _controller!.getMinExposureOffset();

      final clampedOffset = offset.clamp(minOffset, maxOffset);
      await _controller!.setExposureOffset(clampedOffset);
    } catch (e) {
      throw CameraServiceException('Failed to set exposure offset: $e');
    }
  }

  // 카메라 설정 정보 가져오기
  Future<CameraInfo> getCameraInfo() async {
    if (!_isInitialized || _controller == null) {
      throw CameraServiceException('Camera not initialized');
    }

    try {
      final maxZoom = await _controller!.getMaxZoomLevel();
      final minZoom = await _controller!.getMinZoomLevel();
      final currentZoom = await _controller!.getZoomLevel();
      final maxExposure = await _controller!.getMaxExposureOffset();
      final minExposure = await _controller!.getMinExposureOffset();

      return CameraInfo(
        maxZoomLevel: maxZoom,
        minZoomLevel: minZoom,
        currentZoomLevel: currentZoom,
        maxExposureOffset: maxExposure,
        minExposureOffset: minExposure,
        resolutionPreset: ResolutionPreset.high,
        lensDirection: _controller!.description.lensDirection,
        sensorOrientation: _controller!.description.sensorOrientation,
      );
    } catch (e) {
      throw CameraServiceException('Failed to get camera info: $e');
    }
  }

  @override
  Future<void> dispose() async {
    _stopFrameStreaming();
    await _frameStreamController?.close();
    _frameStreamController = null;

    if (_controller != null) {
      await _controller!.dispose();
      _controller = null;
    }

    _isInitialized = false;
    _isRecording = false;
    print('🗑️ Camera service disposed');
  }
}

/// 카메라 정보 모델
class CameraInfo {
  final double maxZoomLevel;
  final double minZoomLevel;
  final double currentZoomLevel;
  final double maxExposureOffset;
  final double minExposureOffset;
  final ResolutionPreset resolutionPreset;
  final CameraLensDirection lensDirection;
  final int sensorOrientation;

  const CameraInfo({
    required this.maxZoomLevel,
    required this.minZoomLevel,
    required this.currentZoomLevel,
    required this.maxExposureOffset,
    required this.minExposureOffset,
    required this.resolutionPreset,
    required this.lensDirection,
    required this.sensorOrientation,
  });
}

/// 카메라 서비스 예외 클래스
class CameraServiceException implements Exception {
  final String message;
  CameraServiceException(this.message);

  @override
  String toString() => 'CameraServiceException: $message';
}