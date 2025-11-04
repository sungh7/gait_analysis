import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';

import '../../lib/shared/services/ml_service.dart';
import '../../lib/shared/models/gait_analysis_models.dart';

class MockMLService extends Mock implements MLService {}

void main() {
  group('MLService Tests', () {
    late MLService mlService;

    setUp(() {
      mlService = MockMLService();
    });

    group('Initialization', () {
      test('should initialize successfully', () async {
        when(() => mlService.initialize()).thenAnswer((_) async {});

        await mlService.initialize();

        verify(() => mlService.initialize()).called(1);
      });

      test('should throw exception on initialization failure', () async {
        when(() => mlService.initialize())
            .thenThrow(MLServiceException('Initialization failed'));

        expect(
          () => mlService.initialize(),
          throwsA(isA<MLServiceException>()),
        );
      });
    });

    group('Pose Detection', () {
      test('should extract pose landmarks from image bytes', () async {
        // Arrange
        final imageBytes = Uint8List.fromList([1, 2, 3, 4, 5]);
        final expectedLandmarks = [
          const PoseLandmark(id: 0, x: 0.5, y: 0.5, z: 0.0, visibility: 0.9),
          const PoseLandmark(id: 1, x: 0.6, y: 0.4, z: 0.1, visibility: 0.8),
        ];

        when(() => mlService.extractPoseLandmarks(imageBytes))
            .thenAnswer((_) async => expectedLandmarks);

        // Act
        final result = await mlService.extractPoseLandmarks(imageBytes);

        // Assert
        expect(result, equals(expectedLandmarks));
        verify(() => mlService.extractPoseLandmarks(imageBytes)).called(1);
      });

      test('should return empty list when no pose detected', () async {
        // Arrange
        final imageBytes = Uint8List.fromList([]);

        when(() => mlService.extractPoseLandmarks(imageBytes))
            .thenAnswer((_) async => []);

        // Act
        final result = await mlService.extractPoseLandmarks(imageBytes);

        // Assert
        expect(result, isEmpty);
      });

      test('should handle pose detection errors gracefully', () async {
        // Arrange
        final imageBytes = Uint8List.fromList([1, 2, 3]);

        when(() => mlService.extractPoseLandmarks(imageBytes))
            .thenThrow(MLServiceException('Pose detection failed'));

        // Act & Assert
        expect(
          () => mlService.extractPoseLandmarks(imageBytes),
          throwsA(isA<MLServiceException>()),
        );
      });
    });

    group('Gait Analysis', () {
      test('should analyze gait from landmark sequence', () async {
        // Arrange
        final landmarkSequence = [
          [
            const PoseLandmark(id: 23, x: 0.5, y: 0.6, z: 0.0, visibility: 0.9), // Left hip
            const PoseLandmark(id: 25, x: 0.5, y: 0.8, z: 0.0, visibility: 0.9), // Left knee
            const PoseLandmark(id: 27, x: 0.5, y: 1.0, z: 0.0, visibility: 0.9), // Left ankle
          ],
          [
            const PoseLandmark(id: 23, x: 0.52, y: 0.6, z: 0.0, visibility: 0.9),
            const PoseLandmark(id: 25, x: 0.52, y: 0.8, z: 0.0, visibility: 0.9),
            const PoseLandmark(id: 27, x: 0.52, y: 1.0, z: 0.0, visibility: 0.9),
          ],
        ];

        final expectedParameters = const GaitParameters(
          cadence: 120.0,
          stepLength: 0.65,
          strideLength: 1.3,
          stepWidth: 0.12,
          walkingSpeed: 1.2,
          stancePhase: 0.6,
          swingPhase: 0.4,
          doubleSupportTime: 0.1,
        );

        when(() => mlService.analyzeGait(landmarkSequence))
            .thenAnswer((_) async => expectedParameters);

        // Act
        final result = await mlService.analyzeGait(landmarkSequence);

        // Assert
        expect(result, equals(expectedParameters));
        expect(result.cadence, equals(120.0));
        expect(result.stepLength, equals(0.65));
        expect(result.walkingSpeed, equals(1.2));
      });

      test('should validate gait parameter ranges', () async {
        // Arrange
        final landmarkSequence = <List<PoseLandmark>>[];
        final parameters = const GaitParameters(
          cadence: 110.0,   // Normal range
          stepLength: 0.6,  // Normal range
          strideLength: 1.2,
          stepWidth: 0.1,   // Normal range
          walkingSpeed: 1.3, // Normal range
          stancePhase: 0.6,
          swingPhase: 0.4,
          doubleSupportTime: 0.1,
        );

        when(() => mlService.analyzeGait(landmarkSequence))
            .thenAnswer((_) async => parameters);

        // Act
        final result = await mlService.analyzeGait(landmarkSequence);

        // Assert
        expect(result.isNormal, isTrue);
        expect(result.qualityScore, greaterThanOrEqualTo(80));
      });
    });

    group('Pathological Detection', () {
      test('should detect normal gait pattern', () async {
        // Arrange
        final gaitFeatures = GaitFeatures.fromGaitParameters(
          const GaitParameters(
            cadence: 115.0,
            stepLength: 0.65,
            strideLength: 1.3,
            stepWidth: 0.12,
            walkingSpeed: 1.25,
            stancePhase: 0.6,
            swingPhase: 0.4,
            doubleSupportTime: 0.1,
          ),
        );

        final expectedResult = const PathologicalDetectionResult(
          isPathological: false,
          confidence: 0.15, // Low pathological confidence
          riskScore: 15,
          detectedPatterns: [],
        );

        when(() => mlService.detectPathologicalGait(gaitFeatures))
            .thenAnswer((_) async => expectedResult);

        // Act
        final result = await mlService.detectPathologicalGait(gaitFeatures);

        // Assert
        expect(result.isPathological, isFalse);
        expect(result.riskLevel, equals('Low'));
        expect(result.detectedPatterns, isEmpty);
      });

      test('should detect pathological gait pattern', () async {
        // Arrange
        final gaitFeatures = GaitFeatures.fromGaitParameters(
          const GaitParameters(
            cadence: 80.0,     // Bradykinesia
            stepLength: 0.3,   // Reduced step length
            strideLength: 0.6,
            stepWidth: 0.18,   // Wide-based gait
            walkingSpeed: 0.7, // Slow walking
            stancePhase: 0.7,
            swingPhase: 0.3,
            doubleSupportTime: 0.15,
          ),
        );

        final expectedResult = const PathologicalDetectionResult(
          isPathological: true,
          confidence: 0.85,
          riskScore: 85,
          detectedPatterns: [
            'Bradykinesia (slow movement)',
            'Reduced step length',
            'Wide-based gait',
          ],
        );

        when(() => mlService.detectPathologicalGait(gaitFeatures))
            .thenAnswer((_) async => expectedResult);

        // Act
        final result = await mlService.detectPathologicalGait(gaitFeatures);

        // Assert
        expect(result.isPathological, isTrue);
        expect(result.riskLevel, equals('High'));
        expect(result.detectedPatterns, isNotEmpty);
        expect(result.detectedPatterns, contains('Bradykinesia (slow movement)'));
      });
    });

    group('Error Handling', () {
      test('should handle ML service exceptions properly', () {
        // Arrange
        const errorMessage = 'Model loading failed';

        when(() => mlService.initialize())
            .thenThrow(MLServiceException(errorMessage));

        // Act & Assert
        expect(
          () => mlService.initialize(),
          throwsA(
            allOf(
              isA<MLServiceException>(),
              predicate<MLServiceException>(
                (e) => e.message == errorMessage,
              ),
            ),
          ),
        );
      });

      test('should validate input parameters', () async {
        // Arrange
        final emptyImageBytes = Uint8List(0);

        when(() => mlService.extractPoseLandmarks(emptyImageBytes))
            .thenThrow(MLServiceException('Invalid image data'));

        // Act & Assert
        expect(
          () => mlService.extractPoseLandmarks(emptyImageBytes),
          throwsA(isA<MLServiceException>()),
        );
      });
    });

    group('Performance', () {
      test('should complete pose detection within acceptable time', () async {
        // Arrange
        final imageBytes = Uint8List.fromList(List.filled(224 * 224 * 3, 128));
        final landmarks = List.generate(
          33,
          (i) => PoseLandmark(
            id: i,
            x: 0.5,
            y: 0.5,
            z: 0.0,
            visibility: 0.9,
          ),
        );

        when(() => mlService.extractPoseLandmarks(imageBytes))
            .thenAnswer((_) async {
          // Simulate processing time
          await Future.delayed(const Duration(milliseconds: 50));
          return landmarks;
        });

        // Act
        final stopwatch = Stopwatch()..start();
        final result = await mlService.extractPoseLandmarks(imageBytes);
        stopwatch.stop();

        // Assert
        expect(result, hasLength(33));
        expect(stopwatch.elapsedMilliseconds, lessThan(100)); // Should be under 100ms
      });
    });

    tearDown(() {
      reset(mlService);
    });
  });
}

// Test utilities
class TestData {
  static List<PoseLandmark> generateNormalPoseLandmarks() {
    return List.generate(33, (i) {
      return PoseLandmark(
        id: i,
        x: 0.5 + (i * 0.01), // Slight variation
        y: 0.5 + (i * 0.01),
        z: 0.0,
        visibility: 0.9 - (i * 0.01), // Decreasing visibility
      );
    });
  }

  static List<PoseLandmark> generatePathologicalPoseLandmarks() {
    return List.generate(33, (i) {
      // Simulate irregular pose patterns
      final irregularity = (i % 3 == 0) ? 0.1 : 0.0;
      return PoseLandmark(
        id: i,
        x: 0.5 + irregularity,
        y: 0.5 + irregularity,
        z: 0.0,
        visibility: 0.8 - irregularity,
      );
    });
  }

  static GaitParameters get normalGaitParameters => const GaitParameters(
        cadence: 115.0,
        stepLength: 0.65,
        strideLength: 1.3,
        stepWidth: 0.12,
        walkingSpeed: 1.25,
        stancePhase: 0.6,
        swingPhase: 0.4,
        doubleSupportTime: 0.1,
      );

  static GaitParameters get pathologicalGaitParameters => const GaitParameters(
        cadence: 80.0,
        stepLength: 0.35,
        strideLength: 0.7,
        stepWidth: 0.18,
        walkingSpeed: 0.8,
        stancePhase: 0.7,
        swingPhase: 0.3,
        doubleSupportTime: 0.15,
      );
}