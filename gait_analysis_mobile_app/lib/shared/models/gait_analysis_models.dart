import 'package:equatable/equatable.dart';
import 'package:json_annotation/json_annotation.dart';

part 'gait_analysis_models.g.dart';

/// 포즈 랜드마크 모델
@JsonSerializable()
class PoseLandmark extends Equatable {
  final int id;
  final double x;
  final double y;
  final double z;
  final double visibility;

  const PoseLandmark({
    required this.id,
    required this.x,
    required this.y,
    required this.z,
    required this.visibility,
  });

  factory PoseLandmark.fromJson(Map<String, dynamic> json) =>
      _$PoseLandmarkFromJson(json);

  Map<String, dynamic> toJson() => _$PoseLandmarkToJson(this);

  @override
  List<Object?> get props => [id, x, y, z, visibility];

  PoseLandmark copyWith({
    int? id,
    double? x,
    double? y,
    double? z,
    double? visibility,
  }) {
    return PoseLandmark(
      id: id ?? this.id,
      x: x ?? this.x,
      y: y ?? this.y,
      z: z ?? this.z,
      visibility: visibility ?? this.visibility,
    );
  }
}

/// 보행 파라미터 모델
@JsonSerializable()
class GaitParameters extends Equatable {
  final double cadence; // steps/min
  final double stepLength; // meters
  final double strideLength; // meters
  final double stepWidth; // meters
  final double walkingSpeed; // m/s
  final double stancePhase; // percentage (0-1)
  final double swingPhase; // percentage (0-1)
  final double doubleSupportTime; // percentage (0-1)

  const GaitParameters({
    required this.cadence,
    required this.stepLength,
    required this.strideLength,
    required this.stepWidth,
    required this.walkingSpeed,
    required this.stancePhase,
    required this.swingPhase,
    required this.doubleSupportTime,
  });

  factory GaitParameters.fromJson(Map<String, dynamic> json) =>
      _$GaitParametersFromJson(json);

  Map<String, dynamic> toJson() => _$GaitParametersToJson(this);

  @override
  List<Object?> get props => [
        cadence,
        stepLength,
        strideLength,
        stepWidth,
        walkingSpeed,
        stancePhase,
        swingPhase,
        doubleSupportTime,
      ];

  /// 보행 파라미터가 정상 범위인지 확인
  bool get isNormal {
    return cadence >= 100 && cadence <= 130 &&
           stepLength >= 0.5 && stepLength <= 0.8 &&
           walkingSpeed >= 1.0 && walkingSpeed <= 1.6;
  }

  /// 보행 품질 점수 (0-100)
  int get qualityScore {
    double score = 100.0;

    // 각 파라미터의 정상 범위에서 벗어날수록 점수 감소
    if (cadence < 100 || cadence > 130) {
      score -= 20;
    }
    if (stepLength < 0.5 || stepLength > 0.8) {
      score -= 20;
    }
    if (walkingSpeed < 1.0 || walkingSpeed > 1.6) {
      score -= 20;
    }
    if (stepWidth < 0.08 || stepWidth > 0.15) {
      score -= 15;
    }
    if (stancePhase < 0.55 || stancePhase > 0.65) {
      score -= 15;
    }
    if (doubleSupportTime < 0.08 || doubleSupportTime > 0.15) {
      score -= 10;
    }

    return score.round().clamp(0, 100);
  }
}

/// 관절 각도 모델
@JsonSerializable()
class JointAngles extends Equatable {
  final double hip;
  final double knee;
  final double ankle;

  const JointAngles({
    required this.hip,
    required this.knee,
    required this.ankle,
  });

  factory JointAngles.fromJson(Map<String, dynamic> json) =>
      _$JointAnglesFromJson(json);

  Map<String, dynamic> toJson() => _$JointAnglesToJson(this);

  @override
  List<Object?> get props => [hip, knee, ankle];
}

/// 프레임 데이터 모델
@JsonSerializable()
class FrameData extends Equatable {
  final int frameNumber;
  final int timestamp; // milliseconds
  final List<PoseLandmark> landmarks;
  final JointAngles leftJointAngles;
  final JointAngles rightJointAngles;

  const FrameData({
    required this.frameNumber,
    required this.timestamp,
    required this.landmarks,
    required this.leftJointAngles,
    required this.rightJointAngles,
  });

  factory FrameData.fromJson(Map<String, dynamic> json) =>
      _$FrameDataFromJson(json);

  Map<String, dynamic> toJson() => _$FrameDataToJson(this);

  @override
  List<Object?> get props => [
        frameNumber,
        timestamp,
        landmarks,
        leftJointAngles,
        rightJointAngles,
      ];
}

/// 보행 분석 결과 모델
@JsonSerializable()
class GaitAnalysisResult extends Equatable {
  final String id;
  final DateTime timestamp;
  final String? patientId;
  final String videoPath;
  final int duration; // seconds
  final int frameCount;
  final int processingTime; // milliseconds
  final GaitParameters gaitParameters;
  final List<FrameData> frames;
  final int qualityScore; // 0-100
  final List<String> recommendations;
  final PathologicalDetectionResult? pathologicalResult;

  const GaitAnalysisResult({
    required this.id,
    required this.timestamp,
    this.patientId,
    required this.videoPath,
    required this.duration,
    required this.frameCount,
    required this.processingTime,
    required this.gaitParameters,
    required this.frames,
    required this.qualityScore,
    required this.recommendations,
    this.pathologicalResult,
  });

  factory GaitAnalysisResult.fromJson(Map<String, dynamic> json) =>
      _$GaitAnalysisResultFromJson(json);

  Map<String, dynamic> toJson() => _$GaitAnalysisResultToJson(this);

  @override
  List<Object?> get props => [
        id,
        timestamp,
        patientId,
        videoPath,
        duration,
        frameCount,
        processingTime,
        gaitParameters,
        frames,
        qualityScore,
        recommendations,
        pathologicalResult,
      ];

  /// 분석 성공 여부
  bool get isSuccessful => qualityScore >= 70 && frames.isNotEmpty;

  /// 처리 속도 (fps)
  double get processingFps => frameCount / (processingTime / 1000);
}

/// 보행 특징 모델 (GAVD 시스템 기반)
@JsonSerializable()
class GaitFeatures extends Equatable {
  final double cadence;
  final double stepLength;
  final double strideLength;
  final double stepWidth;
  final double walkingSpeed;
  final double stancePhase;
  final double swingPhase;
  final double doubleSupportTime;
  final double stepLengthVariability;
  final double stepTimeVariability;
  final double asymmetryIndex;
  final double harmonyIndex;
  final double stabilityIndex;
  final double rhythmIndex;
  final double smoothnessIndex;
  final double coordinationIndex;
  final double balanceIndex;
  final double energyEfficiency;
  final double overallGaitIndex;

  const GaitFeatures({
    required this.cadence,
    required this.stepLength,
    required this.strideLength,
    required this.stepWidth,
    required this.walkingSpeed,
    required this.stancePhase,
    required this.swingPhase,
    required this.doubleSupportTime,
    required this.stepLengthVariability,
    required this.stepTimeVariability,
    required this.asymmetryIndex,
    required this.harmonyIndex,
    required this.stabilityIndex,
    required this.rhythmIndex,
    required this.smoothnessIndex,
    required this.coordinationIndex,
    required this.balanceIndex,
    required this.energyEfficiency,
    required this.overallGaitIndex,
  });

  factory GaitFeatures.fromJson(Map<String, dynamic> json) =>
      _$GaitFeaturesFromJson(json);

  Map<String, dynamic> toJson() => _$GaitFeaturesToJson(this);

  @override
  List<Object?> get props => [
        cadence, stepLength, strideLength, stepWidth, walkingSpeed,
        stancePhase, swingPhase, doubleSupportTime, stepLengthVariability,
        stepTimeVariability, asymmetryIndex, harmonyIndex, stabilityIndex,
        rhythmIndex, smoothnessIndex, coordinationIndex, balanceIndex,
        energyEfficiency, overallGaitIndex,
      ];

  /// 19차원 특징 벡터로 변환 (ML 모델 입력용)
  List<double> toVector() {
    return [
      cadence, stepLength, strideLength, stepWidth, walkingSpeed,
      stancePhase, swingPhase, doubleSupportTime, stepLengthVariability,
      stepTimeVariability, asymmetryIndex, harmonyIndex, stabilityIndex,
      rhythmIndex, smoothnessIndex, coordinationIndex, balanceIndex,
      energyEfficiency, overallGaitIndex,
    ];
  }

  /// GaitParameters로부터 GaitFeatures 생성
  factory GaitFeatures.fromGaitParameters(GaitParameters params) {
    return GaitFeatures(
      cadence: params.cadence,
      stepLength: params.stepLength,
      strideLength: params.strideLength,
      stepWidth: params.stepWidth,
      walkingSpeed: params.walkingSpeed,
      stancePhase: params.stancePhase,
      swingPhase: params.swingPhase,
      doubleSupportTime: params.doubleSupportTime,
      stepLengthVariability: 0.1, // 기본값들 (향후 계산 로직 추가)
      stepTimeVariability: 0.05,
      asymmetryIndex: 0.1,
      harmonyIndex: 0.8,
      stabilityIndex: 0.85,
      rhythmIndex: 0.9,
      smoothnessIndex: 0.8,
      coordinationIndex: 0.85,
      balanceIndex: 0.8,
      energyEfficiency: 0.75,
      overallGaitIndex: 0.8,
    );
  }
}

/// 병적 보행 검출 결과 모델
@JsonSerializable()
class PathologicalDetectionResult extends Equatable {
  final bool isPathological;
  final double confidence; // 0-1
  final int riskScore; // 0-100
  final List<String> detectedPatterns;
  final Map<String, double> patternProbabilities;

  const PathologicalDetectionResult({
    required this.isPathological,
    required this.confidence,
    required this.riskScore,
    required this.detectedPatterns,
    this.patternProbabilities = const {},
  });

  factory PathologicalDetectionResult.fromJson(Map<String, dynamic> json) =>
      _$PathologicalDetectionResultFromJson(json);

  Map<String, dynamic> toJson() => _$PathologicalDetectionResultToJson(this);

  @override
  List<Object?> get props => [
        isPathological,
        confidence,
        riskScore,
        detectedPatterns,
        patternProbabilities,
      ];

  /// 위험도 레벨
  String get riskLevel {
    if (riskScore < 30) return 'Low';
    if (riskScore < 60) return 'Moderate';
    if (riskScore < 80) return 'High';
    return 'Critical';
  }

  /// 권장 사항 생성
  List<String> get recommendations {
    final recommendations = <String>[];

    if (isPathological) {
      recommendations.add('의료진과 상담을 권장합니다.');

      if (detectedPatterns.contains('Bradykinesia')) {
        recommendations.add('파킨슨병 검사를 고려해보세요.');
      }
      if (detectedPatterns.contains('Gait asymmetry')) {
        recommendations.add('편마비 또는 근골격계 이상을 확인하세요.');
      }
      if (detectedPatterns.contains('Wide-based gait')) {
        recommendations.add('소뇌 기능 또는 균형 검사가 필요할 수 있습니다.');
      }
    } else {
      recommendations.add('정상적인 보행 패턴입니다.');
      recommendations.add('규칙적인 운동을 통해 보행 능력을 유지하세요.');
    }

    return recommendations;
  }
}

/// 환자 정보 모델
@JsonSerializable()
class Patient extends Equatable {
  final String id;
  final String name;
  final int age;
  final double height; // cm
  final double weight; // kg
  final String gender;
  final List<String> medicalConditions;
  final DateTime createdAt;
  final DateTime updatedAt;

  const Patient({
    required this.id,
    required this.name,
    required this.age,
    required this.height,
    required this.weight,
    required this.gender,
    required this.medicalConditions,
    required this.createdAt,
    required this.updatedAt,
  });

  factory Patient.fromJson(Map<String, dynamic> json) =>
      _$PatientFromJson(json);

  Map<String, dynamic> toJson() => _$PatientToJson(this);

  @override
  List<Object?> get props => [
        id, name, age, height, weight, gender,
        medicalConditions, createdAt, updatedAt,
      ];

  /// BMI 계산
  double get bmi => weight / ((height / 100) * (height / 100));

  /// BMI 분류
  String get bmiCategory {
    if (bmi < 18.5) return 'Underweight';
    if (bmi < 25) return 'Normal';
    if (bmi < 30) return 'Overweight';
    return 'Obese';
  }
}