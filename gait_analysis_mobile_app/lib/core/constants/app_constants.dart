class AppConstants {
  // App Information
  static const String appName = 'Gait Analysis Pro';
  static const String appVersion = '1.0.0';
  static const String appDescription = 'Enterprise-grade gait analysis using AI';

  // Environment Configuration
  static const String environment = String.fromEnvironment(
    'ENVIRONMENT',
    defaultValue: 'development',
  );

  static const bool isProduction = environment == 'production';
  static const bool isDevelopment = environment == 'development';
  static const bool isStaging = environment == 'staging';

  // API Configuration
  static const String baseUrl = String.fromEnvironment(
    'BASE_URL',
    defaultValue: 'https://api-dev.gaitanalysis.com',
  );

  static const String mlServiceUrl = String.fromEnvironment(
    'ML_SERVICE_URL',
    defaultValue: 'https://ml-dev.gaitanalysis.com',
  );

  // Firebase Configuration
  static const String firebaseProjectId = 'gait-analysis-pro';

  // External Services
  static const String sentryDsn = String.fromEnvironment('SENTRY_DSN', defaultValue: '');

  // ML Model Configuration
  static const String gaitAnalysisModelPath = 'assets/models/gait_analysis_model.tflite';
  static const String poseEstimationModelPath = 'assets/models/pose_estimation_model.tflite';

  // Camera Configuration
  static const int defaultCameraResolutionWidth = 1280;
  static const int defaultCameraResolutionHeight = 720;
  static const int targetFps = 30;
  static const double minConfidenceThreshold = 0.7;

  // Analysis Configuration
  static const int minAnalysisDurationSeconds = 10;
  static const int maxAnalysisDurationSeconds = 60;
  static const int minFramesForAnalysis = 300;

  // Storage Configuration
  static const String videosDirectoryName = 'videos';
  static const String resultsDirectoryName = 'analysis_results';
  static const String modelsDirectoryName = 'ml_models';

  // Network Configuration
  static const int connectionTimeoutMs = 30000;
  static const int receiveTimeoutMs = 30000;
  static const int sendTimeoutMs = 30000;

  // UI Configuration
  static const double defaultPadding = 16.0;
  static const double defaultBorderRadius = 8.0;
  static const double defaultElevation = 2.0;

  // Animation Durations
  static const int shortAnimationDuration = 200;
  static const int mediumAnimationDuration = 400;
  static const int longAnimationDuration = 600;

  // Pagination
  static const int defaultPageSize = 20;
  static const int maxPageSize = 100;

  // File Size Limits
  static const int maxVideoFileSizeMB = 500;
  static const int maxImageFileSizeMB = 10;

  // Cache Configuration
  static const int cacheExpirationHours = 24;
  static const int maxCacheSize = 100; // Number of items

  // Error Messages
  static const String genericErrorMessage = 'An unexpected error occurred. Please try again.';
  static const String networkErrorMessage = 'Network error. Please check your connection.';
  static const String cameraPermissionErrorMessage = 'Camera permission is required for gait analysis.';
  static const String storagePermissionErrorMessage = 'Storage permission is required to save analysis results.';

  // Success Messages
  static const String analysisCompletedMessage = 'Gait analysis completed successfully.';
  static const String videoSavedMessage = 'Video saved successfully.';
  static const String dataExportedMessage = 'Data exported successfully.';

  // Feature Flags
  static const bool enableOfflineMode = true;
  static const bool enableCloudSync = true;
  static const bool enableAdvancedAnalytics = true;
  static const bool enableBetaFeatures = isDevelopment;

  // Medical Data Configuration
  static const List<String> supportedFileFormats = ['.mp4', '.mov', '.avi'];
  static const List<String> supportedImageFormats = ['.jpg', '.jpeg', '.png'];

  // Regulatory Compliance
  static const bool hipaaCompliant = true;
  static const bool gdprCompliant = true;
  static const String privacyPolicyUrl = 'https://gaitanalysis.com/privacy';
  static const String termsOfServiceUrl = 'https://gaitanalysis.com/terms';

  // Research & Development
  static const bool enableDataCollection = isProduction;
  static const bool enableTelemetry = true;
  static const String researchDataEndpoint = '/api/v1/research/data';
}