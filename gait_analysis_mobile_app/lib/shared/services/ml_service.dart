import 'dart:typed_data';
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

import '../../core/constants/app_constants.dart';
import '../models/gait_analysis_models.dart';

/// ML 서비스 추상 클래스
abstract class MLService {
  Future<void> initialize();
  Future<List<PoseLandmark>> extractPoseLandmarks(Uint8List imageBytes);
  Future<GaitParameters> analyzeGait(List<List<PoseLandmark>> landmarkSequence);
  Future<PathologicalDetectionResult> detectPathologicalGait(GaitFeatures features);
  Future<void> dispose();
}

/// TensorFlow Lite 기반 ML 서비스 구현
class MLServiceImpl implements MLService {
  Interpreter? _poseEstimationInterpreter;
  Interpreter? _gaitAnalysisInterpreter;
  Interpreter? _pathologicalDetectionInterpreter;

  bool _isInitialized = false;

  @override
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      // 포즈 추정 모델 로드
      _poseEstimationInterpreter = await _loadModel(
        AppConstants.poseEstimationModelPath,
      );

      // 보행 분석 모델 로드
      _gaitAnalysisInterpreter = await _loadModel(
        AppConstants.gaitAnalysisModelPath,
      );

      // 병적 보행 검출 모델 로드
      _pathologicalDetectionInterpreter = await _loadModel(
        'assets/models/pathological_detection_model.tflite',
      );

      _isInitialized = true;
      print('✅ ML Service initialized successfully');
    } catch (e) {
      print('❌ Failed to initialize ML Service: $e');
      throw MLServiceException('Failed to initialize ML models: $e');
    }
  }

  Future<Interpreter> _loadModel(String modelPath) async {
    try {
      // 에셋에서 모델 파일 로드
      final modelBytes = await rootBundle.load(modelPath);
      final modelBuffer = modelBytes.buffer.asUint8List();

      // TensorFlow Lite 인터프리터 생성
      final options = InterpreterOptions();

      // GPU 델리게이트 사용 (가능한 경우)
      if (Platform.isAndroid) {
        options.addDelegate(GpuDelegate());
      }

      return Interpreter.fromBuffer(modelBuffer, options: options);
    } catch (e) {
      throw MLServiceException('Failed to load model $modelPath: $e');
    }
  }

  @override
  Future<List<PoseLandmark>> extractPoseLandmarks(Uint8List imageBytes) async {
    if (!_isInitialized) await initialize();

    try {
      // 이미지 전처리
      final processedImage = _preprocessImage(imageBytes);

      // 모델 입력 설정
      final input = [processedImage];
      final output = [List.generate(33, (_) => List.filled(4, 0.0))];

      // 추론 실행
      _poseEstimationInterpreter!.run(input, output);

      // 결과를 PoseLandmark 리스트로 변환
      final landmarks = <PoseLandmark>[];
      for (int i = 0; i < 33; i++) {
        landmarks.add(PoseLandmark(
          id: i,
          x: output[0][i][0],
          y: output[0][i][1],
          z: output[0][i][2],
          visibility: output[0][i][3],
        ));
      }

      return landmarks;
    } catch (e) {
      throw MLServiceException('Failed to extract pose landmarks: $e');
    }
  }

  List<List<List<double>>> _preprocessImage(Uint8List imageBytes) {
    // 이미지 디코딩
    final image = img.decodeImage(imageBytes);
    if (image == null) {
      throw MLServiceException('Failed to decode image');
    }

    // 224x224로 리사이즈
    final resized = img.copyResize(
      image,
      width: 224,
      height: 224,
      interpolation: img.Interpolation.linear,
    );

    // 정규화 (0-255 -> 0-1)
    final input = List.generate(
      224,
      (y) => List.generate(
        224,
        (x) => [
          img.getRed(resized, x, y) / 255.0,
          img.getGreen(resized, x, y) / 255.0,
          img.getBlue(resized, x, y) / 255.0,
        ],
      ),
    );

    return input;
  }

  @override
  Future<GaitParameters> analyzeGait(
    List<List<PoseLandmark>> landmarkSequence,
  ) async {
    if (!_isInitialized) await initialize();

    try {
      // 시퀀스 길이 조정 (100 프레임으로 패딩 또는 자르기)
      final processedSequence = _preprocessLandmarkSequence(landmarkSequence);

      // 모델 입력 설정
      final input = [processedSequence];
      final output = [List.filled(5, 0.0)];

      // 추론 실행
      _gaitAnalysisInterpreter!.run(input, output);

      // 결과를 GaitParameters로 변환
      return GaitParameters(
        cadence: output[0][0] * 180, // 0-1 범위를 실제 값으로 스케일링
        stepLength: output[0][1] * 2.0, // 미터 단위
        strideLength: output[0][2] * 4.0,
        stepWidth: output[0][3] * 0.5,
        walkingSpeed: output[0][4] * 3.0, // m/s
        stancePhase: 0.6, // 기본값 (향후 모델 확장)
        swingPhase: 0.4,
        doubleSupportTime: 0.1,
      );
    } catch (e) {
      throw MLServiceException('Failed to analyze gait: $e');
    }
  }

  List<List<List<List<double>>>> _preprocessLandmarkSequence(
    List<List<PoseLandmark>> sequence,
  ) {
    const targetLength = 100;

    // 시퀀스 길이 조정
    List<List<PoseLandmark>> adjustedSequence;
    if (sequence.length < targetLength) {
      // 패딩 (마지막 프레임 반복)
      adjustedSequence = List.from(sequence);
      while (adjustedSequence.length < targetLength) {
        adjustedSequence.add(sequence.last);
      }
    } else {
      // 균등하게 샘플링
      adjustedSequence = [];
      for (int i = 0; i < targetLength; i++) {
        final index = (i * sequence.length / targetLength).floor();
        adjustedSequence.add(sequence[index]);
      }
    }

    // 4차원 텐서로 변환: [batch, sequence, landmarks, features]
    return [
      adjustedSequence.map((frame) =>
        frame.map((landmark) => [
          landmark.x,
          landmark.y,
          landmark.z,
          landmark.visibility,
        ]).toList()
      ).toList()
    ];
  }

  @override
  Future<PathologicalDetectionResult> detectPathologicalGait(
    GaitFeatures features,
  ) async {
    if (!_isInitialized) await initialize();

    try {
      // 특징 벡터를 모델 입력 형식으로 변환
      final input = [features.toVector()];
      final output = [List.filled(1, 0.0)];

      // 추론 실행
      _pathologicalDetectionInterpreter!.run(input, output);

      final probability = output[0][0];
      final isPathological = probability > 0.5;

      return PathologicalDetectionResult(
        isPathological: isPathological,
        confidence: probability,
        riskScore: (probability * 100).round(),
        detectedPatterns: _analyzePathologicalPatterns(features, probability),
      );
    } catch (e) {
      throw MLServiceException('Failed to detect pathological gait: $e');
    }
  }

  List<String> _analyzePathologicalPatterns(
    GaitFeatures features,
    double probability,
  ) {
    final patterns = <String>[];

    // 보행 패턴 분석 로직
    if (features.cadence < 100) {
      patterns.add('Bradykinesia (slow movement)');
    }
    if (features.stepLength < 0.4) {
      patterns.add('Reduced step length');
    }
    if (features.stepWidth > 0.15) {
      patterns.add('Wide-based gait');
    }
    if (features.asymmetryIndex > 0.2) {
      patterns.add('Gait asymmetry');
    }

    return patterns;
  }

  @override
  Future<void> dispose() async {
    _poseEstimationInterpreter?.close();
    _gaitAnalysisInterpreter?.close();
    _pathologicalDetectionInterpreter?.close();

    _poseEstimationInterpreter = null;
    _gaitAnalysisInterpreter = null;
    _pathologicalDetectionInterpreter = null;

    _isInitialized = false;
  }
}

/// ML 서비스 예외 클래스
class MLServiceException implements Exception {
  final String message;
  MLServiceException(this.message);

  @override
  String toString() => 'MLServiceException: $message';
}

/// GPU 델리게이트 (Android 전용)
class GpuDelegate extends Delegate {
  GpuDelegate() : super._(GpuDelegateType());
}

class GpuDelegateType extends DelegateType {
  @override
  String get name => 'GPU';
}