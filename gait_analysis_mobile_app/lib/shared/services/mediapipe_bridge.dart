import 'dart:typed_data';
import 'package:flutter/services.dart';
import '../models/gait_analysis_models.dart';

/// MediaPipe Native Bridge
///
/// Flutter와 네이티브 MediaPipe 코드 간의 통신 브릿지
class MediaPipeBridge {
  static const MethodChannel _channel =
      MethodChannel('gait_analysis/mediapipe');

  bool _isInitialized = false;

  /// MediaPipe 초기화
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      final result = await _channel.invokeMethod('initialize');
      print('✅ MediaPipe initialized: $result');
      _isInitialized = true;
    } on PlatformException catch (e) {
      print('❌ Failed to initialize MediaPipe: ${e.message}');
      throw MediaPipeException('Initialization failed: ${e.message}');
    }
  }

  /// 단일 이미지에서 Pose 검출
  Future<List<PoseLandmark>> detectPose(
    Uint8List imageBytes, {
    int? timestampMs,
  }) async {
    if (!_isInitialized) {
      throw MediaPipeException('MediaPipe not initialized');
    }

    try {
      final result = await _channel.invokeMethod('detectPose', {
        'imageBytes': imageBytes,
        'timestampMs': timestampMs ?? DateTime.now().millisecondsSinceEpoch,
      });

      if (result == null) {
        return [];
      }

      final landmarks = (result['landmarks'] as List)
          .map((lm) => PoseLandmark.fromJson(Map<String, dynamic>.from(lm)))
          .toList();

      return landmarks;
    } on PlatformException catch (e) {
      print('❌ Pose detection error: ${e.message}');
      throw MediaPipeException('Detection failed: ${e.message}');
    }
  }

  /// 비디오 모드에서 Pose 검출 (최적화됨)
  Future<List<PoseLandmark>> detectPoseVideo(
    Uint8List imageBytes, {
    required int timestampMs,
  }) async {
    if (!_isInitialized) {
      throw MediaPipeException('MediaPipe not initialized');
    }

    try {
      final result = await _channel.invokeMethod('detectPoseVideo', {
        'imageBytes': imageBytes,
        'timestampMs': timestampMs,
      });

      if (result == null) {
        return [];
      }

      // World landmarks를 우선 사용 (3D 좌표)
      final worldLandmarks = result['worldLandmarks'] as List?;
      if (worldLandmarks != null && worldLandmarks.isNotEmpty) {
        return worldLandmarks
            .map((lm) => PoseLandmark.fromJson(Map<String, dynamic>.from(lm)))
            .toList();
      }

      // Fallback to normalized landmarks
      final landmarks = (result['landmarks'] as List)
          .map((lm) => PoseLandmark.fromJson(Map<String, dynamic>.from(lm)))
          .toList();

      return landmarks;
    } on PlatformException catch (e) {
      print('❌ Video pose detection error: ${e.message}');
      throw MediaPipeException('Video detection failed: ${e.message}');
    }
  }

  /// MediaPipe 정리
  Future<void> dispose() async {
    if (!_isInitialized) return;

    try {
      await _channel.invokeMethod('dispose');
      _isInitialized = false;
      print('✅ MediaPipe disposed');
    } on PlatformException catch (e) {
      print('❌ Failed to dispose MediaPipe: ${e.message}');
    }
  }

  /// 초기화 상태 확인
  bool get isInitialized => _isInitialized;
}

/// MediaPipe 예외
class MediaPipeException implements Exception {
  final String message;
  MediaPipeException(this.message);

  @override
  String toString() => 'MediaPipeException: $message';
}

/// PoseLandmark 확장 (JSON 파싱)
extension PoseLandmarkJson on PoseLandmark {
  static PoseLandmark fromJson(Map<String, dynamic> json) {
    return PoseLandmark(
      id: json['id'] as int,
      position: json['position'] as String? ?? 'unknown',
      x: (json['worldX'] as num?)?.toDouble() ?? 0.0,
      y: (json['worldY'] as num?)?.toDouble() ?? 0.0,
      z: (json['worldZ'] as num?)?.toDouble() ?? 0.0,
      visibility: (json['visibility'] as num?)?.toDouble() ?? 1.0,
    );
  }
}
