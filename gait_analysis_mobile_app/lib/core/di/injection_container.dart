import 'package:get_it/get_it.dart';
import 'package:dio/dio.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_firestore/firebase_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';

import '../constants/app_constants.dart';
import '../network/api_client.dart';
import '../network/network_interceptor.dart';
import '../storage/storage_service.dart';
import '../../shared/repositories/auth_repository.dart';
import '../../shared/repositories/gait_analysis_repository.dart';
import '../../shared/repositories/patient_repository.dart';
import '../../shared/services/ml_service.dart';
import '../../shared/services/camera_service.dart';
import '../../shared/services/video_processing_service.dart';
import '../../features/authentication/data/datasources/auth_remote_datasource.dart';
import '../../features/authentication/data/repositories/auth_repository_impl.dart';
import '../../features/authentication/domain/usecases/login_usecase.dart';
import '../../features/authentication/domain/usecases/logout_usecase.dart';
import '../../features/authentication/presentation/bloc/auth_bloc.dart';

final sl = GetIt.instance;

Future<void> init() async {
  // External dependencies
  await _initExternalDependencies();

  // Core dependencies
  _initCore();

  // Repository dependencies
  _initRepositories();

  // Use case dependencies
  _initUseCases();

  // BLoC dependencies
  _initBlocs();

  // Service dependencies
  _initServices();
}

Future<void> _initExternalDependencies() async {
  // SharedPreferences
  final sharedPreferences = await SharedPreferences.getInstance();
  sl.registerLazySingleton(() => sharedPreferences);

  // Firebase instances
  sl.registerLazySingleton(() => FirebaseAuth.instance);
  sl.registerLazySingleton(() => FirebaseFirestore.instance);
  sl.registerLazySingleton(() => FirebaseStorage.instance);

  // Dio HTTP client
  final dio = Dio();
  dio.options.baseUrl = AppConstants.baseUrl;
  dio.options.connectTimeout = const Duration(milliseconds: AppConstants.connectionTimeoutMs);
  dio.options.receiveTimeout = const Duration(milliseconds: AppConstants.receiveTimeoutMs);
  dio.options.sendTimeout = const Duration(milliseconds: AppConstants.sendTimeoutMs);

  // Add interceptors
  dio.interceptors.add(NetworkInterceptor());

  if (!AppConstants.isProduction) {
    dio.interceptors.add(LogInterceptor(
      requestBody: true,
      responseBody: true,
      requestHeader: true,
      responseHeader: true,
    ));
  }

  sl.registerLazySingleton(() => dio);
}

void _initCore() {
  // API Client
  sl.registerLazySingleton<ApiClient>(() => ApiClient(sl()));

  // Storage Service
  sl.registerLazySingleton<StorageService>(() => StorageService(sl()));
}

void _initRepositories() {
  // Authentication Repository
  sl.registerLazySingleton<AuthRepository>(
    () => AuthRepositoryImpl(
      remoteDataSource: sl(),
      storageService: sl(),
    ),
  );

  // Gait Analysis Repository
  sl.registerLazySingleton<GaitAnalysisRepository>(
    () => GaitAnalysisRepositoryImpl(
      apiClient: sl(),
      storageService: sl(),
      mlService: sl(),
    ),
  );

  // Patient Repository
  sl.registerLazySingleton<PatientRepository>(
    () => PatientRepositoryImpl(
      firestore: sl(),
      storageService: sl(),
    ),
  );
}

void _initUseCases() {
  // Authentication Use Cases
  sl.registerLazySingleton(() => LoginUseCase(sl()));
  sl.registerLazySingleton(() => LogoutUseCase(sl()));

  // Gait Analysis Use Cases
  sl.registerLazySingleton(() => AnalyzeGaitUseCase(sl()));
  sl.registerLazySingleton(() => GetAnalysisHistoryUseCase(sl()));
  sl.registerLazySingleton(() => ExportAnalysisResultUseCase(sl()));

  // Patient Use Cases
  sl.registerLazySingleton(() => GetPatientsUseCase(sl()));
  sl.registerLazySingleton(() => CreatePatientUseCase(sl()));
  sl.registerLazySingleton(() => UpdatePatientUseCase(sl()));
  sl.registerLazySingleton(() => DeletePatientUseCase(sl()));
}

void _initBlocs() {
  // Authentication BLoC
  sl.registerFactory(
    () => AuthBloc(
      loginUseCase: sl(),
      logoutUseCase: sl(),
    ),
  );

  // Gait Analysis BLoC
  sl.registerFactory(
    () => GaitAnalysisBloc(
      analyzeGaitUseCase: sl(),
      getAnalysisHistoryUseCase: sl(),
      exportAnalysisResultUseCase: sl(),
    ),
  );

  // Camera BLoC
  sl.registerFactory(
    () => CameraBloc(
      cameraService: sl(),
      videoProcessingService: sl(),
    ),
  );

  // Patient BLoC
  sl.registerFactory(
    () => PatientBloc(
      getPatientsUseCase: sl(),
      createPatientUseCase: sl(),
      updatePatientUseCase: sl(),
      deletePatientUseCase: sl(),
    ),
  );
}

void _initServices() {
  // ML Service
  sl.registerLazySingleton<MLService>(() => MLServiceImpl());

  // Camera Service
  sl.registerLazySingleton<CameraService>(() => CameraServiceImpl());

  // Video Processing Service
  sl.registerLazySingleton<VideoProcessingService>(
    () => VideoProcessingServiceImpl(
      mlService: sl(),
    ),
  );

  // Data Sources
  sl.registerLazySingleton<AuthRemoteDataSource>(
    () => AuthRemoteDataSourceImpl(
      firebaseAuth: sl(),
      apiClient: sl(),
    ),
  );
}