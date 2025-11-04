import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'dart:isolate';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:image/image.dart' as img;

import '../../core/constants/app_constants.dart';
import '../models/gait_analysis_models.dart';
import 'ml_service.dart';

/// 비디오 처리 서비스 추상 클래스
abstract class VideoProcessingService {
  Future<GaitAnalysisResult> processVideo(String videoPath, {String? patientId});
  Future<List<PoseLandmark>> processFrame(Uint8List frameData);
  Stream<VideoProcessingProgress> processVideoStream(String videoPath);
  Future<void> cancelProcessing();
}

/// 비디오 처리 진행 상태
class VideoProcessingProgress {
  final int currentFrame;
  final int totalFrames;
  final double percentage;
  final String status;
  final List<PoseLandmark>? currentLandmarks;

  const VideoProcessingProgress({
    required this.currentFrame,
    required this.totalFrames,
    required this.percentage,
    required this.status,
    this.currentLandmarks,
  });

  VideoProcessingProgress copyWith({
    int? currentFrame,
    int? totalFrames,
    double? percentage,
    String? status,
    List<PoseLandmark>? currentLandmarks,
  }) {
    return VideoProcessingProgress(
      currentFrame: currentFrame ?? this.currentFrame,
      totalFrames: totalFrames ?? this.totalFrames,
      percentage: percentage ?? this.percentage,
      status: status ?? this.status,
      currentLandmarks: currentLandmarks ?? this.currentLandmarks,
    );
  }
}

/// 비디오 처리 서비스 구현
class VideoProcessingServiceImpl implements VideoProcessingService {
  final MLService _mlService;
  StreamController<VideoProcessingProgress>? _progressController;
  Isolate? _processingIsolate;
  bool _isProcessing = false;
  bool _isCancelled = false;

  VideoProcessingServiceImpl({required MLService mlService})
      : _mlService = mlService;

  @override
  Future<GaitAnalysisResult> processVideo(
    String videoPath, {
    String? patientId,
  }) async {
    if (_isProcessing) {
      throw VideoProcessingException('Already processing a video');
    }

    _isProcessing = true;
    _isCancelled = false;

    try {
      // ML 서비스 초기화
      await _mlService.initialize();

      final startTime = DateTime.now();
      print('🎬 Starting video processing: $videoPath');

      // 비디오 메타데이터 추출
      final videoInfo = await _extractVideoInfo(videoPath);
      print('📊 Video info: ${videoInfo.duration}s, ${videoInfo.frameCount} frames');

      // 프레임 추출 및 처리
      final frames = await _extractAndProcessFrames(videoPath, videoInfo);

      if (_isCancelled) {
        throw VideoProcessingException('Processing cancelled');
      }

      // 보행 분석 수행
      final gaitParameters = await _analyzeGaitFromFrames(frames);

      // 병적 보행 검출
      final gaitFeatures = GaitFeatures.fromGaitParameters(gaitParameters);
      final pathologicalResult = await _mlService.detectPathologicalGait(gaitFeatures);

      // 결과 생성
      final processingTime = DateTime.now().difference(startTime);
      final result = GaitAnalysisResult(
        id: _generateResultId(),
        timestamp: DateTime.now(),
        patientId: patientId,
        videoPath: videoPath,
        duration: videoInfo.duration,
        frameCount: videoInfo.frameCount,
        processingTime: processingTime.inMilliseconds,
        gaitParameters: gaitParameters,
        frames: frames,
        qualityScore: _calculateQualityScore(frames, gaitParameters),
        recommendations: _generateRecommendations(gaitParameters, pathologicalResult),
        pathologicalResult: pathologicalResult,
      );

      print('✅ Video processing completed: ${processingTime.inSeconds}s');
      return result;
    } catch (e) {
      print('❌ Video processing failed: $e');
      throw VideoProcessingException('Failed to process video: $e');
    } finally {
      _isProcessing = false;
      _isCancelled = false;
    }
  }

  @override
  Stream<VideoProcessingProgress> processVideoStream(String videoPath) {
    if (_progressController != null) {
      _progressController!.close();
    }

    _progressController = StreamController<VideoProcessingProgress>.broadcast();
    _processVideoInBackground(videoPath);

    return _progressController!.stream;
  }

  Future<void> _processVideoInBackground(String videoPath) async {
    try {
      await _mlService.initialize();
      final videoInfo = await _extractVideoInfo(videoPath);

      _progressController?.add(VideoProcessingProgress(
        currentFrame: 0,
        totalFrames: videoInfo.frameCount,
        percentage: 0.0,
        status: 'Extracting frames...',
      ));

      final frames = <FrameData>[];
      int frameIndex = 0;

      // 프레임별 처리 (실시간 피드백)
      await for (final frameData in _extractFramesStream(videoPath)) {
        if (_isCancelled) break;

        try {
          // 포즈 랜드마크 추출
          final landmarks = await _mlService.extractPoseLandmarks(frameData);

          // 관절 각도 계산
          final leftAngles = _calculateJointAngles(landmarks, isLeft: true);
          final rightAngles = _calculateJointAngles(landmarks, isRight: true);

          final frame = FrameData(
            frameNumber: frameIndex,
            timestamp: (frameIndex * 1000 / videoInfo.fps).round(),
            landmarks: landmarks,
            leftJointAngles: leftAngles,
            rightJointAngles: rightAngles,
          );

          frames.add(frame);

          // 진행 상태 업데이트
          final progress = (frameIndex + 1) / videoInfo.frameCount;
          _progressController?.add(VideoProcessingProgress(
            currentFrame: frameIndex + 1,
            totalFrames: videoInfo.frameCount,
            percentage: progress * 0.8, // 80%는 프레임 처리
            status: 'Processing frame ${frameIndex + 1}/${videoInfo.frameCount}',
            currentLandmarks: landmarks,
          ));

          frameIndex++;
        } catch (e) {
          print('Frame processing error at $frameIndex: $e');
          // 에러가 발생한 프레임은 건너뛰기
        }
      }

      if (!_isCancelled && frames.isNotEmpty) {
        // 보행 분석 수행
        _progressController?.add(VideoProcessingProgress(
          currentFrame: frameIndex,
          totalFrames: videoInfo.frameCount,
          percentage: 0.9,
          status: 'Analyzing gait parameters...',
        ));

        final gaitParameters = await _analyzeGaitFromFrames(frames);
        final gaitFeatures = GaitFeatures.fromGaitParameters(gaitParameters);
        final pathologicalResult = await _mlService.detectPathologicalGait(gaitFeatures);

        // 완료
        _progressController?.add(VideoProcessingProgress(
          currentFrame: frameIndex,
          totalFrames: videoInfo.frameCount,
          percentage: 1.0,
          status: 'Analysis completed',
        ));
      }
    } catch (e) {
      _progressController?.addError(VideoProcessingException('Processing failed: $e'));
    } finally {
      _progressController?.close();
      _progressController = null;
    }
  }

  @override
  Future<List<PoseLandmark>> processFrame(Uint8List frameData) async {
    await _mlService.initialize();
    return await _mlService.extractPoseLandmarks(frameData);
  }

  @override
  Future<void> cancelProcessing() async {
    _isCancelled = true;
    _processingIsolate?.kill();
    _processingIsolate = null;
    _progressController?.close();
    _progressController = null;
    print('🛑 Video processing cancelled');
  }

  // 비디오 정보 추출
  Future<VideoInfo> _extractVideoInfo(String videoPath) async {
    // FFmpeg를 사용하거나 플랫폼별 API를 사용하여 비디오 정보 추출
    // 여기서는 기본값으로 설정
    return VideoInfo(
      path: videoPath,
      duration: 30, // 초
      frameCount: 900, // 30fps * 30초
      fps: 30,
      width: 1280,
      height: 720,
    );
  }

  // 프레임 추출 및 처리
  Future<List<FrameData>> _extractAndProcessFrames(
    String videoPath,
    VideoInfo videoInfo,
  ) async {
    final frames = <FrameData>[];

    // 실제 구현에서는 FFmpeg나 플랫폼별 API 사용
    // 여기서는 시뮬레이션
    for (int i = 0; i < videoInfo.frameCount && !_isCancelled; i++) {
      try {
        // 더미 프레임 데이터 (실제로는 비디오에서 추출)
        final dummyFrameData = _generateDummyFrameData();
        final landmarks = await _mlService.extractPoseLandmarks(dummyFrameData);

        final leftAngles = _calculateJointAngles(landmarks, isLeft: true);
        final rightAngles = _calculateJointAngles(landmarks, isRight: true);

        final frame = FrameData(
          frameNumber: i,
          timestamp: (i * 1000 / videoInfo.fps).round(),
          landmarks: landmarks,
          leftJointAngles: leftAngles,
          rightJointAngles: rightAngles,
        );

        frames.add(frame);

        // 진행 상태 업데이트
        if (i % 10 == 0) {
          print('Processed frame $i/${videoInfo.frameCount}');
        }
      } catch (e) {
        print('Error processing frame $i: $e');
      }
    }

    return frames;
  }

  // 프레임 스트림 추출
  Stream<Uint8List> _extractFramesStream(String videoPath) async* {
    // 실제 구현에서는 비디오 파일에서 프레임을 순차적으로 추출
    // 여기서는 시뮬레이션
    for (int i = 0; i < 100; i++) {
      if (_isCancelled) break;
      yield _generateDummyFrameData();
      await Future.delayed(const Duration(milliseconds: 33)); // ~30fps
    }
  }

  // 더미 프레임 데이터 생성 (테스트용)
  Uint8List _generateDummyFrameData() {
    // 224x224 RGB 더미 이미지 생성
    final image = img.Image(width: 224, height: 224);
    img.fill(image, color: img.ColorRgb8(128, 128, 128));
    return Uint8List.fromList(img.encodeJpg(image));
  }

  // 보행 분석
  Future<GaitParameters> _analyzeGaitFromFrames(List<FrameData> frames) async {
    if (frames.isEmpty) {
      throw VideoProcessingException('No frames to analyze');
    }

    // 랜드마크 시퀀스 준비
    final landmarkSequence = frames.map((frame) => frame.landmarks).toList();

    // ML 서비스를 사용하여 보행 파라미터 추출
    return await _mlService.analyzeGait(landmarkSequence);
  }

  // 관절 각도 계산
  JointAngles _calculateJointAngles(
    List<PoseLandmark> landmarks, {
    bool isLeft = false,
    bool isRight = false,
  }) {
    // MediaPipe 포즈 랜드마크 인덱스 기반 각도 계산
    // 실제 구현에서는 3D 벡터 계산 사용

    if (isLeft) {
      // 왼쪽 관절 각도 계산
      return const JointAngles(
        hip: 170.0,   // 고관절 각도
        knee: 160.0,  // 슬관절 각도
        ankle: 90.0,  // 족관절 각도
      );
    } else {
      // 오른쪽 관절 각도 계산
      return const JointAngles(
        hip: 165.0,
        knee: 155.0,
        ankle: 95.0,
      );
    }
  }

  // 품질 점수 계산
  int _calculateQualityScore(List<FrameData> frames, GaitParameters parameters) {
    if (frames.isEmpty) return 0;

    // 랜드마크 가시성 기반 품질 계산
    double totalVisibility = 0.0;
    int landmarkCount = 0;

    for (final frame in frames) {
      for (final landmark in frame.landmarks) {
        totalVisibility += landmark.visibility;
        landmarkCount++;
      }
    }

    final averageVisibility = landmarkCount > 0 ? totalVisibility / landmarkCount : 0.0;
    final frameCompleteness = frames.length / 300.0; // 최소 300프레임 기준

    // 보행 파라미터 품질도 고려
    final parameterQuality = parameters.qualityScore / 100.0;

    final overallQuality = (averageVisibility * 0.4 +
        frameCompleteness.clamp(0.0, 1.0) * 0.3 +
        parameterQuality * 0.3);

    return (overallQuality * 100).round().clamp(0, 100);
  }

  // 권장사항 생성
  List<String> _generateRecommendations(
    GaitParameters parameters,
    PathologicalDetectionResult pathologicalResult,
  ) {
    final recommendations = <String>[];

    // 병적 보행 결과 기반 권장사항
    recommendations.addAll(pathologicalResult.recommendations);

    // 보행 파라미터 기반 권장사항
    if (parameters.cadence < 100) {
      recommendations.add('보행 속도를 증가시키는 운동을 권장합니다.');
    }
    if (parameters.stepLength < 0.5) {
      recommendations.add('보폭을 늘리는 연습이 필요합니다.');
    }
    if (parameters.stepWidth > 0.15) {
      recommendations.add('균형 감각을 향상시키는 운동을 권장합니다.');
    }

    return recommendations;
  }

  String _generateResultId() {
    return 'analysis_${DateTime.now().millisecondsSinceEpoch}';
  }
}

/// 비디오 정보 모델
class VideoInfo {
  final String path;
  final int duration; // seconds
  final int frameCount;
  final double fps;
  final int width;
  final int height;

  const VideoInfo({
    required this.path,
    required this.duration,
    required this.frameCount,
    required this.fps,
    required this.width,
    required this.height,
  });
}

/// 비디오 처리 예외 클래스
class VideoProcessingException implements Exception {
  final String message;
  VideoProcessingException(this.message);

  @override
  String toString() => 'VideoProcessingException: $message';
}