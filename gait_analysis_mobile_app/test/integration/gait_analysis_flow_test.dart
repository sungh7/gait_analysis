import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:gait_analysis_app/main.dart' as app;

import '../helpers/test_helpers.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('🧪 Complete Gait Analysis Flow Integration Tests', () {
    testWidgets('Complete gait analysis workflow', (WidgetTester tester) async {
      // Initialize app
      app.main();
      await tester.pumpAndSettle();

      // Test 1: App Launch and Initialization
      await _testAppLaunch(tester);

      // Test 2: User Authentication
      await _testUserAuthentication(tester);

      // Test 3: Camera Setup and Permissions
      await _testCameraSetup(tester);

      // Test 4: Video Recording
      await _testVideoRecording(tester);

      // Test 5: Gait Analysis Processing
      await _testGaitAnalysisProcessing(tester);

      // Test 6: Results Display
      await _testResultsDisplay(tester);

      // Test 7: Data Export
      await _testDataExport(tester);

      // Test 8: History and Patient Management
      await _testHistoryAndPatientManagement(tester);
    });

    testWidgets('Error handling and recovery', (WidgetTester tester) async {
      app.main();
      await tester.pumpAndSettle();

      // Test error scenarios
      await _testNetworkErrorHandling(tester);
      await _testCameraErrorHandling(tester);
      await _testProcessingErrorHandling(tester);
    });

    testWidgets('Performance benchmarks', (WidgetTester tester) async {
      app.main();
      await tester.pumpAndSettle();

      // Test performance metrics
      await _testAppStartupTime(tester);
      await _testVideoProcessingPerformance(tester);
      await _testMemoryUsage(tester);
    });

    testWidgets('Accessibility compliance', (WidgetTester tester) async {
      app.main();
      await tester.pumpAndSettle();

      // Test accessibility features
      await _testScreenReader(tester);
      await _testKeyboardNavigation(tester);
      await _testContrastRatios(tester);
    });
  });
}

// ========================================
// Test Helper Functions
// ========================================

Future<void> _testAppLaunch(WidgetTester tester) async {
  group('App Launch', () {
    test('should display splash screen', () async {
      expect(find.byKey(const Key('splash_screen')), findsOneWidget);
      await tester.pumpAndSettle(const Duration(seconds: 3));
    });

    test('should navigate to home screen', () async {
      expect(find.byKey(const Key('home_screen')), findsOneWidget);
      expect(find.text('Gait Analysis Pro'), findsOneWidget);
    });

    test('should initialize services', () async {
      // Verify that ML service is initialized
      final mlServiceStatus = find.byKey(const Key('ml_service_status'));
      expect(mlServiceStatus, findsOneWidget);

      // Verify camera service is ready
      final cameraStatus = find.byKey(const Key('camera_service_status'));
      expect(cameraStatus, findsOneWidget);
    });
  });
}

Future<void> _testUserAuthentication(WidgetTester tester) async {
  group('User Authentication', () {
    test('should handle guest mode', () async {
      final guestButton = find.byKey(const Key('guest_mode_button'));
      await tester.tap(guestButton);
      await tester.pumpAndSettle();

      expect(find.byKey(const Key('main_app')), findsOneWidget);
    });

    test('should authenticate with credentials', () async {
      // Navigate to login
      final loginButton = find.byKey(const Key('login_button'));
      await tester.tap(loginButton);
      await tester.pumpAndSettle();

      // Enter test credentials
      await tester.enterText(
        find.byKey(const Key('email_field')),
        'test@gaitanalysis.com',
      );
      await tester.enterText(
        find.byKey(const Key('password_field')),
        'testPassword123',
      );

      // Submit login
      final submitButton = find.byKey(const Key('submit_login'));
      await tester.tap(submitButton);
      await tester.pumpAndSettle();

      // Verify successful login
      expect(find.byKey(const Key('user_profile')), findsOneWidget);
    });
  });
}

Future<void> _testCameraSetup(WidgetTester tester) async {
  group('Camera Setup', () {
    test('should request camera permissions', () async {
      // Navigate to camera screen
      final cameraTab = find.byKey(const Key('camera_tab'));
      await tester.tap(cameraTab);
      await tester.pumpAndSettle();

      // Grant camera permission (simulated)
      await TestHelpers.simulatePermissionGrant(Permission.camera);

      // Verify camera preview is displayed
      expect(find.byKey(const Key('camera_preview')), findsOneWidget);
    });

    test('should configure camera settings', () async {
      // Open camera settings
      final settingsButton = find.byKey(const Key('camera_settings'));
      await tester.tap(settingsButton);
      await tester.pumpAndSettle();

      // Adjust resolution
      final resolutionSlider = find.byKey(const Key('resolution_slider'));
      await tester.drag(resolutionSlider, const Offset(100, 0));
      await tester.pumpAndSettle();

      // Apply settings
      final applyButton = find.byKey(const Key('apply_settings'));
      await tester.tap(applyButton);
      await tester.pumpAndSettle();
    });
  });
}

Future<void> _testVideoRecording(WidgetTester tester) async {
  group('Video Recording', () {
    test('should start and stop recording', () async {
      // Start recording
      final recordButton = find.byKey(const Key('record_button'));
      await tester.tap(recordButton);
      await tester.pumpAndSettle();

      // Verify recording indicator
      expect(find.byKey(const Key('recording_indicator')), findsOneWidget);

      // Wait for minimum recording duration
      await tester.pump(const Duration(seconds: 10));

      // Stop recording
      await tester.tap(recordButton);
      await tester.pumpAndSettle();

      // Verify recording stopped
      expect(find.byKey(const Key('recording_indicator')), findsNothing);
      expect(find.byKey(const Key('video_preview')), findsOneWidget);
    });

    test('should handle recording controls', () async {
      // Test pause/resume functionality
      await tester.tap(find.byKey(const Key('record_button')));
      await tester.pumpAndSettle();

      final pauseButton = find.byKey(const Key('pause_button'));
      await tester.tap(pauseButton);
      await tester.pump(const Duration(seconds: 2));

      final resumeButton = find.byKey(const Key('resume_button'));
      await tester.tap(resumeButton);
      await tester.pump(const Duration(seconds: 5));

      await tester.tap(find.byKey(const Key('stop_button')));
      await tester.pumpAndSettle();
    });
  });
}

Future<void> _testGaitAnalysisProcessing(WidgetTester tester) async {
  group('Gait Analysis Processing', () {
    test('should process video for gait analysis', () async {
      // Start analysis
      final analyzeButton = find.byKey(const Key('analyze_button'));
      await tester.tap(analyzeButton);
      await tester.pumpAndSettle();

      // Verify processing starts
      expect(find.byKey(const Key('processing_indicator')), findsOneWidget);

      // Wait for processing to complete
      await TestHelpers.waitForCondition(
        () => find.byKey(const Key('processing_complete')).evaluate().isNotEmpty,
        timeout: const Duration(minutes: 2),
      );

      // Verify processing completed
      expect(find.byKey(const Key('analysis_results')), findsOneWidget);
    });

    test('should show real-time processing progress', () async {
      // Verify progress indicators
      expect(find.byKey(const Key('frame_counter')), findsOneWidget);
      expect(find.byKey(const Key('progress_bar')), findsOneWidget);
      expect(find.byKey(const Key('processing_status')), findsOneWidget);

      // Verify pose landmarks visualization
      expect(find.byKey(const Key('pose_overlay')), findsOneWidget);
    });
  });
}

Future<void> _testResultsDisplay(WidgetTester tester) async {
  group('Results Display', () {
    test('should display gait parameters', () async {
      // Verify parameter cards
      expect(find.byKey(const Key('cadence_card')), findsOneWidget);
      expect(find.byKey(const Key('step_length_card')), findsOneWidget);
      expect(find.byKey(const Key('walking_speed_card')), findsOneWidget);

      // Verify parameter values are reasonable
      final cadenceText = TestHelpers.getWidgetText(
        tester,
        find.byKey(const Key('cadence_value')),
      );
      final cadence = double.tryParse(cadenceText);
      expect(cadence, greaterThan(50));
      expect(cadence, lessThan(200));
    });

    test('should display pathological detection results', () async {
      // Navigate to pathological results
      final pathologyTab = find.byKey(const Key('pathology_tab'));
      await tester.tap(pathologyTab);
      await tester.pumpAndSettle();

      // Verify risk assessment
      expect(find.byKey(const Key('risk_score')), findsOneWidget);
      expect(find.byKey(const Key('detected_patterns')), findsOneWidget);
      expect(find.byKey(const Key('recommendations')), findsOneWidget);
    });

    test('should display quality metrics', () async {
      // Navigate to quality tab
      final qualityTab = find.byKey(const Key('quality_tab'));
      await tester.tap(qualityTab);
      await tester.pumpAndSettle();

      // Verify quality indicators
      expect(find.byKey(const Key('overall_quality')), findsOneWidget);
      expect(find.byKey(const Key('landmark_quality')), findsOneWidget);
      expect(find.byKey(const Key('processing_metrics')), findsOneWidget);
    });
  });
}

Future<void> _testDataExport(WidgetTester tester) async {
  group('Data Export', () {
    test('should export results to PDF', () async {
      final exportButton = find.byKey(const Key('export_pdf_button'));
      await tester.tap(exportButton);
      await tester.pumpAndSettle();

      // Verify export success
      expect(find.text('PDF exported successfully'), findsOneWidget);
    });

    test('should export data to CSV', () async {
      final exportButton = find.byKey(const Key('export_csv_button'));
      await tester.tap(exportButton);
      await tester.pumpAndSettle();

      // Verify export success
      expect(find.text('CSV exported successfully'), findsOneWidget);
    });

    test('should share results', () async {
      final shareButton = find.byKey(const Key('share_button'));
      await tester.tap(shareButton);
      await tester.pumpAndSettle();

      // Verify share dialog appears
      expect(find.byKey(const Key('share_dialog')), findsOneWidget);
    });
  });
}

Future<void> _testHistoryAndPatientManagement(WidgetTester tester) async {
  group('History and Patient Management', () {
    test('should display analysis history', () async {
      // Navigate to history
      final historyTab = find.byKey(const Key('history_tab'));
      await tester.tap(historyTab);
      await tester.pumpAndSettle();

      // Verify history list
      expect(find.byKey(const Key('history_list')), findsOneWidget);
    });

    test('should create and manage patients', () async {
      // Navigate to patients
      final patientsTab = find.byKey(const Key('patients_tab'));
      await tester.tap(patientsTab);
      await tester.pumpAndSettle();

      // Create new patient
      final addPatientButton = find.byKey(const Key('add_patient_button'));
      await tester.tap(addPatientButton);
      await tester.pumpAndSettle();

      // Fill patient information
      await tester.enterText(
        find.byKey(const Key('patient_name')),
        'Test Patient',
      );
      await tester.enterText(
        find.byKey(const Key('patient_age')),
        '45',
      );

      // Save patient
      final saveButton = find.byKey(const Key('save_patient'));
      await tester.tap(saveButton);
      await tester.pumpAndSettle();

      // Verify patient created
      expect(find.text('Test Patient'), findsOneWidget);
    });
  });
}

// ========================================
// Error Handling Tests
// ========================================

Future<void> _testNetworkErrorHandling(WidgetTester tester) async {
  group('Network Error Handling', () {
    test('should handle network connectivity issues', () async {
      // Simulate network disconnection
      await TestHelpers.simulateNetworkDisconnection();

      // Attempt to sync data
      final syncButton = find.byKey(const Key('sync_button'));
      await tester.tap(syncButton);
      await tester.pumpAndSettle();

      // Verify error message
      expect(find.text('Network connection error'), findsOneWidget);

      // Restore connection
      await TestHelpers.simulateNetworkReconnection();

      // Retry sync
      final retryButton = find.byKey(const Key('retry_button'));
      await tester.tap(retryButton);
      await tester.pumpAndSettle();

      // Verify success
      expect(find.text('Sync completed'), findsOneWidget);
    });
  });
}

Future<void> _testCameraErrorHandling(WidgetTester tester) async {
  group('Camera Error Handling', () {
    test('should handle camera permission denied', () async {
      // Simulate permission denial
      await TestHelpers.simulatePermissionDenial(Permission.camera);

      // Navigate to camera
      final cameraTab = find.byKey(const Key('camera_tab'));
      await tester.tap(cameraTab);
      await tester.pumpAndSettle();

      // Verify permission request dialog
      expect(find.byKey(const Key('permission_dialog')), findsOneWidget);
    });
  });
}

Future<void> _testProcessingErrorHandling(WidgetTester tester) async {
  group('Processing Error Handling', () {
    test('should handle ML model loading errors', () async {
      // Simulate model loading failure
      await TestHelpers.simulateMLModelError();

      // Attempt analysis
      final analyzeButton = find.byKey(const Key('analyze_button'));
      await tester.tap(analyzeButton);
      await tester.pumpAndSettle();

      // Verify error handling
      expect(find.text('Analysis failed'), findsOneWidget);
      expect(find.byKey(const Key('retry_analysis')), findsOneWidget);
    });
  });
}

// ========================================
// Performance Tests
// ========================================

Future<void> _testAppStartupTime(WidgetTester tester) async {
  group('Performance - App Startup', () {
    test('should start within acceptable time', () async {
      final stopwatch = Stopwatch()..start();

      // App initialization
      app.main();
      await tester.pumpAndSettle();

      stopwatch.stop();

      // Assert startup time is under 3 seconds
      expect(stopwatch.elapsedMilliseconds, lessThan(3000));
    });
  });
}

Future<void> _testVideoProcessingPerformance(WidgetTester tester) async {
  group('Performance - Video Processing', () {
    test('should process video at acceptable rate', () async {
      final stopwatch = Stopwatch()..start();

      // Start processing
      final analyzeButton = find.byKey(const Key('analyze_button'));
      await tester.tap(analyzeButton);

      // Wait for completion
      await TestHelpers.waitForCondition(
        () => find.byKey(const Key('processing_complete')).evaluate().isNotEmpty,
        timeout: const Duration(minutes: 2),
      );

      stopwatch.stop();

      // Assert processing time is reasonable (< 2 minutes for 30 second video)
      expect(stopwatch.elapsedMilliseconds, lessThan(120000));
    });
  });
}

Future<void> _testMemoryUsage(WidgetTester tester) async {
  group('Performance - Memory Usage', () {
    test('should maintain reasonable memory usage', () async {
      // Get initial memory usage
      final initialMemory = await TestHelpers.getMemoryUsage();

      // Perform multiple analyses
      for (int i = 0; i < 5; i++) {
        await _simulateVideoAnalysis(tester);
      }

      // Check final memory usage
      final finalMemory = await TestHelpers.getMemoryUsage();

      // Memory increase should be reasonable (< 200MB)
      final memoryIncrease = finalMemory - initialMemory;
      expect(memoryIncrease, lessThan(200 * 1024 * 1024)); // 200MB
    });
  });
}

// ========================================
// Accessibility Tests
// ========================================

Future<void> _testScreenReader(WidgetTester tester) async {
  group('Accessibility - Screen Reader', () {
    test('should have proper semantic labels', () async {
      // Check main navigation elements
      expect(
        find.bySemanticsLabel('Camera for gait analysis'),
        findsOneWidget,
      );
      expect(
        find.bySemanticsLabel('Analysis results'),
        findsOneWidget,
      );
      expect(
        find.bySemanticsLabel('Patient history'),
        findsOneWidget,
      );
    });
  });
}

Future<void> _testKeyboardNavigation(WidgetTester tester) async {
  group('Accessibility - Keyboard Navigation', () {
    test('should support tab navigation', () async {
      // Test tab navigation through main elements
      await tester.sendKeyEvent(LogicalKeyboardKey.tab);
      await tester.pumpAndSettle();

      // Verify focus moves to next element
      expect(
        find.byKey(const Key('focused_element')),
        findsOneWidget,
      );
    });
  });
}

Future<void> _testContrastRatios(WidgetTester tester) async {
  group('Accessibility - Contrast Ratios', () {
    test('should meet WCAG contrast requirements', () async {
      // This would typically use a specialized testing framework
      // for color contrast validation
      final contrastResults = await TestHelpers.checkContrastRatios(tester);
      expect(contrastResults.meetsWCAGAA, isTrue);
    });
  });
}

// ========================================
// Helper Functions
// ========================================

Future<void> _simulateVideoAnalysis(WidgetTester tester) async {
  final analyzeButton = find.byKey(const Key('analyze_button'));
  await tester.tap(analyzeButton);
  await TestHelpers.waitForCondition(
    () => find.byKey(const Key('processing_complete')).evaluate().isNotEmpty,
    timeout: const Duration(minutes: 1),
  );
}