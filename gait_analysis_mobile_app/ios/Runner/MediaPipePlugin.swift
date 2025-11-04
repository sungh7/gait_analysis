import Flutter
import UIKit
import MediaPipeTasksVision
import AVFoundation
import Accelerate

/**
 * MediaPipe 네이티브 플러그인 - iOS 구현
 * 고성능 포즈 추정을 위한 iOS 네이티브 구현
 */
public class MediaPipePlugin: NSObject, FlutterPlugin {

    static let channelName = "gait_analysis/mediapipe"

    private var poseDetector: PoseLandmarker?
    private var isInitialized = false

    // 성능 메트릭
    private var inferenceTimeHistory: [Double] = []
    private var frameCount: Int64 = 0
    private var startTime = CFAbsoluteTimeGetCurrent()

    // 백그라운드 큐
    private let processingQueue = DispatchQueue(
        label: "com.gaitanalysis.mediapipe.processing",
        qos: .userInitiated,
        attributes: .concurrent
    )

    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: channelName,
            binaryMessenger: registrar.messenger()
        )
        let instance = MediaPipePlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {
        case "initialize":
            initialize(call: call, result: result)
        case "detectPose":
            detectPose(call: call, result: result)
        case "detectPoseVideo":
            detectPoseVideo(call: call, result: result)
        case "getPerformanceMetrics":
            getPerformanceMetrics(result: result)
        case "dispose":
            dispose(result: result)
        default:
            result(FlutterMethodNotImplemented)
        }
    }

    private func initialize(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard let args = call.arguments as? [String: Any] else {
            result([
                "success": false,
                "error": "Invalid arguments"
            ])
            return
        }

        let enableGpu = args["enableGpu"] as? Bool ?? true
        let confidence = args["confidence"] as? Double ?? 0.7

        do {
            // MediaPipe 모델 경로 설정
            guard let modelPath = Bundle.main.path(
                forResource: "pose_landmarker",
                ofType: "task"
            ) else {
                result([
                    "success": false,
                    "error": "Model file not found"
                ])
                return
            }

            // PoseLandmarker 옵션 설정
            let options = PoseLandmarkerOptions()
            options.baseOptions.modelAssetPath = modelPath
            options.runningMode = .video
            options.minPoseDetectionConfidence = Float(confidence)
            options.minPosePresenceConfidence = Float(confidence)
            options.minTrackingConfidence = Float(confidence)
            options.outputSegmentationMasks = false

            // GPU 델리게이트 설정
            if enableGpu {
                options.baseOptions.delegate = .GPU
            }

            // PoseLandmarker 초기화
            poseDetector = try PoseLandmarker(options: options)
            isInitialized = true

            result([
                "success": true,
                "message": "MediaPipe initialized successfully"
            ])

        } catch {
            result([
                "success": false,
                "error": "Failed to initialize MediaPipe: \(error.localizedDescription)"
            ])
        }
    }

    private func detectPose(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard isInitialized, let poseDetector = poseDetector else {
            result([
                "success": false,
                "error": "MediaPipe not initialized"
            ])
            return
        }

        guard let args = call.arguments as? [String: Any],
              let imageData = args["imageBytes"] as? FlutterStandardTypedData,
              let width = args["width"] as? Int,
              let height = args["height"] as? Int else {
            result([
                "success": false,
                "error": "Invalid arguments"
            ])
            return
        }

        // 백그라운드에서 처리
        processingQueue.async { [weak self] in
            self?.processImageAsync(
                imageData: imageData.data,
                width: width,
                height: height,
                poseDetector: poseDetector,
                timestampMs: nil,
                result: result
            )
        }
    }

    private func detectPoseVideo(call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard isInitialized, let poseDetector = poseDetector else {
            result([
                "success": false,
                "error": "MediaPipe not initialized"
            ])
            return
        }

        guard let args = call.arguments as? [String: Any],
              let imageData = args["imageBytes"] as? FlutterStandardTypedData,
              let timestampMs = args["timestampMs"] as? Int else {
            result([
                "success": false,
                "error": "Invalid arguments for video detection"
            ])
            return
        }

        // 백그라운드에서 처리
        processingQueue.async { [weak self] in
            self?.processImageAsync(
                imageData: imageData.data,
                width: nil,
                height: nil,
                poseDetector: poseDetector,
                timestampMs: timestampMs,
                result: result
            )
        }
    }

    private func processImageAsync(
        imageData: Data,
        width: Int?,
        height: Int?,
        poseDetector: PoseLandmarker,
        timestampMs: Int?,
        result: @escaping FlutterResult
    ) {
        let startTime = CFAbsoluteTimeGetCurrent()

        do {
            // Data를 UIImage로 변환
            guard let uiImage = UIImage(data: imageData) else {
                DispatchQueue.main.async {
                    result([
                        "success": false,
                        "error": "Failed to create UIImage from data"
                    ])
                }
                return
            }

            // 크기 조정 (필요한 경우)
            let finalImage: UIImage
            if let width = width, let height = height {
                finalImage = uiImage.resized(to: CGSize(width: width, height: height))
            } else {
                finalImage = uiImage
            }

            // MPImage로 변환
            let mpImage = try MPImage(uiImage: finalImage)

            // 타임스탬프 설정
            let timestamp = timestampMs ?? Int(CFAbsoluteTimeGetCurrent() * 1000)

            // 포즈 검출 실행
            let detectionResult = try poseDetector.detect(
                videoFrame: mpImage,
                timestampInMilliseconds: timestamp
            )

            // 추론 시간 계산
            let inferenceTime = CFAbsoluteTimeGetCurrent() - startTime
            self.updatePerformanceMetrics(inferenceTime: inferenceTime)

            // 결과 변환
            let landmarks = self.convertLandmarks(detectionResult.landmarks)
            let worldLandmarks = self.convertWorldLandmarks(detectionResult.worldLandmarks)

            DispatchQueue.main.async {
                result([
                    "success": true,
                    "landmarks": landmarks,
                    "worldLandmarks": worldLandmarks
                ])
            }

        } catch {
            DispatchQueue.main.async {
                result([
                    "success": false,
                    "error": "Detection failed: \(error.localizedDescription)"
                ])
            }
        }
    }

    private func convertLandmarks(_ landmarksList: [[NormalizedLandmark]]) -> [[String: Any]] {
        guard let landmarks = landmarksList.first else { return [] }

        return landmarks.enumerated().map { index, landmark in
            [
                "id": index,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z ?? 0.0,
                "visibility": landmark.visibility ?? 1.0
            ]
        }
    }

    private func convertWorldLandmarks(_ landmarksList: [[Landmark]]) -> [[String: Any]] {
        guard let landmarks = landmarksList.first else { return [] }

        return landmarks.enumerated().map { index, landmark in
            [
                "id": index,
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility ?? 1.0
            ]
        }
    }

    private func updatePerformanceMetrics(inferenceTime: Double) {
        frameCount += 1

        // 최근 100개 프레임의 추론 시간 유지
        inferenceTimeHistory.append(inferenceTime * 1000) // 밀리초로 변환
        if inferenceTimeHistory.count > 100 {
            inferenceTimeHistory.removeFirst()
        }
    }

    private func getPerformanceMetrics(result: @escaping FlutterResult) {
        let currentTime = CFAbsoluteTimeGetCurrent()
        let elapsedTime = currentTime - startTime
        let fps = elapsedTime > 0 ? Double(frameCount) / elapsedTime : 0.0

        let avgInferenceTime = inferenceTimeHistory.isEmpty
            ? 0.0
            : inferenceTimeHistory.reduce(0, +) / Double(inferenceTimeHistory.count)

        // 메모리 사용량
        let memoryUsage = getMemoryUsage()

        // CPU 사용량
        let cpuUsage = getCpuUsage()

        result([
            "averageInferenceTime": avgInferenceTime,
            "framesPerSecond": fps,
            "memoryUsage": memoryUsage,
            "cpuUsage": cpuUsage,
            "gpuUsage": 0.0 // GPU 사용량은 iOS에서 직접 측정하기 어려움
        ])
    }

    private func getMemoryUsage() -> Int64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4

        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count
                )
            }
        }

        return kerr == KERN_SUCCESS ? Int64(info.resident_size) : 0
    }

    private func getCpuUsage() -> Double {
        var info = processor_info_array_t.allocate(capacity: 1)
        var numCpuInfo: mach_msg_type_number_t = 0
        var numCpus: natural_t = 0

        let result = host_processor_info(
            mach_host_self(),
            PROCESSOR_CPU_LOAD_INFO,
            &numCpus,
            &info,
            &numCpuInfo
        )

        guard result == KERN_SUCCESS else { return 0.0 }

        // CPU 사용량 계산 (간단한 추정)
        // 실제 구현에서는 더 정확한 계산 필요
        info.deallocate()
        return 0.0
    }

    private func dispose(result: @escaping FlutterResult) {
        poseDetector = nil
        isInitialized = false

        // 성능 메트릭 초기화
        inferenceTimeHistory.removeAll()
        frameCount = 0
        startTime = CFAbsoluteTimeGetCurrent()

        result(["success": true])
    }
}

// MARK: - UIImage Extensions

extension UIImage {
    func resized(to size: CGSize) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: size)
        return renderer.image { _ in
            self.draw(in: CGRect(origin: .zero, size: size))
        }
    }

    func pixelBuffer() -> CVPixelBuffer? {
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(size.width),
            Int(size.height),
            kCVPixelFormatType_32ARGB,
            attrs,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(buffer, CVPixelBufferLockFlags(rawValue: 0)) }

        let pixelData = CVPixelBufferGetBaseAddress(buffer)
        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()

        guard let context = CGContext(
            data: pixelData,
            width: Int(size.width),
            height: Int(size.height),
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: rgbColorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        ) else {
            return nil
        }

        context.translateBy(x: 0, y: size.height)
        context.scaleBy(x: 1.0, y: -1.0)

        UIGraphicsPushContext(context)
        draw(in: CGRect(origin: .zero, size: size))
        UIGraphicsPopContext()

        return buffer
    }
}

// MARK: - Performance Monitoring

class PerformanceMonitor {
    static let shared = PerformanceMonitor()

    private var cpuUsageHistory: [Double] = []
    private var memoryUsageHistory: [Int64] = []

    private init() {}

    func startMonitoring() {
        Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            self.recordMetrics()
        }
    }

    private func recordMetrics() {
        let cpuUsage = getCurrentCpuUsage()
        let memoryUsage = getCurrentMemoryUsage()

        cpuUsageHistory.append(cpuUsage)
        memoryUsageHistory.append(memoryUsage)

        // 최근 60개 샘플만 유지 (1분)
        if cpuUsageHistory.count > 60 {
            cpuUsageHistory.removeFirst()
        }
        if memoryUsageHistory.count > 60 {
            memoryUsageHistory.removeFirst()
        }
    }

    private func getCurrentCpuUsage() -> Double {
        var info = processor_info_array_t.allocate(capacity: 1)
        var numCpuInfo: mach_msg_type_number_t = 0
        var numCpus: natural_t = 0

        let result = host_processor_info(
            mach_host_self(),
            PROCESSOR_CPU_LOAD_INFO,
            &numCpus,
            &info,
            &numCpuInfo
        )

        defer { info.deallocate() }

        guard result == KERN_SUCCESS else { return 0.0 }

        // 간단한 CPU 사용량 추정
        return Double.random(in: 10...30) // 실제 구현 필요
    }

    private func getCurrentMemoryUsage() -> Int64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4

        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count
                )
            }
        }

        return kerr == KERN_SUCCESS ? Int64(info.resident_size) : 0
    }

    func getAverageMetrics() -> (cpu: Double, memory: Int64) {
        let avgCpu = cpuUsageHistory.isEmpty
            ? 0.0
            : cpuUsageHistory.reduce(0, +) / Double(cpuUsageHistory.count)

        let avgMemory = memoryUsageHistory.isEmpty
            ? 0
            : memoryUsageHistory.reduce(0, +) / Int64(memoryUsageHistory.count)

        return (cpu: avgCpu, memory: avgMemory)
    }
}