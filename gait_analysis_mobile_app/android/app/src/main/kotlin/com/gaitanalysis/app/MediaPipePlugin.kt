package com.gaitanalysis.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.annotation.NonNull
import com.google.mediapipe.framework.MediaPipeException
import com.google.mediapipe.framework.Packet
import com.google.mediapipe.framework.PacketGetter
import com.google.mediapipe.solutioncore.CameraInput
import com.google.mediapipe.solutioncore.SolutionGlSurfaceView
import com.google.mediapipe.solutions.pose.Pose
import com.google.mediapipe.solutions.pose.PoseOptions
import com.google.mediapipe.solutions.pose.PoseResult
import com.google.mediapipe.solutions.pose.PoseLandmark
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import kotlinx.coroutines.*
import java.nio.ByteBuffer
import kotlin.system.measureTimeMillis

/**
 * MediaPipe 네이티브 플러그인
 * 고성능 포즈 추정을 위한 Android 네이티브 구현
 */
class MediaPipePlugin: FlutterPlugin, MethodCallHandler {
    companion object {
        private const val CHANNEL = "gait_analysis/mediapipe"
        private const val TAG = "MediaPipePlugin"
    }

    private lateinit var channel: MethodChannel
    private lateinit var context: Context

    private var pose: Pose? = null
    private var isInitialized = false

    // 성능 메트릭
    private val inferenceTimeHistory = mutableListOf<Long>()
    private var frameCount = 0L
    private var startTime = System.currentTimeMillis()

    // 스레드 풀
    private val processingScope = CoroutineScope(
        SupervisorJob() + Dispatchers.Default +
        CoroutineName("MediaPipeProcessing")
    )

    override fun onAttachedToEngine(@NonNull flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(flutterPluginBinding.binaryMessenger, CHANNEL)
        channel.setMethodCallHandler(this)
        context = flutterPluginBinding.applicationContext
    }

    override fun onMethodCall(@NonNull call: MethodCall, @NonNull result: Result) {
        when (call.method) {
            "initialize" -> initialize(call, result)
            "detectPose" -> detectPose(call, result)
            "getPerformanceMetrics" -> getPerformanceMetrics(result)
            "dispose" -> dispose(result)
            else -> result.notImplemented()
        }
    }

    private fun initialize(call: MethodCall, result: Result) {
        try {
            val enableGpu = call.argument<Boolean>("enableGpu") ?: true
            val confidence = call.argument<Double>("confidence") ?: 0.7

            // MediaPipe Pose 솔루션 초기화
            val poseOptions = PoseOptions.builder()
                .setStaticImageMode(false)
                .setModelComplexity(2) // 높은 정확도 모델
                .setSmoothLandmarks(true)
                .setEnableSegmentation(false)
                .setMinDetectionConfidence(confidence.toFloat())
                .setMinTrackingConfidence(confidence.toFloat())
                .build()

            pose = Pose(context, poseOptions)

            // 결과 리스너 설정
            pose?.setResultListener { poseResult ->
                // 결과는 detectPose에서 동기적으로 처리
            }

            // 에러 리스너 설정
            pose?.setErrorListener { message, exception ->
                android.util.Log.e(TAG, "MediaPipe Error: $message", exception)
            }

            isInitialized = true

            result.success(mapOf(
                "success" to true,
                "message" to "MediaPipe initialized successfully"
            ))

        } catch (e: Exception) {
            result.success(mapOf(
                "success" to false,
                "error" to "Failed to initialize MediaPipe: ${e.message}"
            ))
        }
    }

    private fun detectPose(call: MethodCall, result: Result) {
        if (!isInitialized || pose == null) {
            result.success(mapOf(
                "success" to false,
                "error" to "MediaPipe not initialized"
            ))
            return
        }

        try {
            val imageBytes = call.argument<ByteArray>("imageBytes")
            val width = call.argument<Int>("width") ?: 224
            val height = call.argument<Int>("height") ?: 224

            if (imageBytes == null) {
                result.success(mapOf(
                    "success" to false,
                    "error" to "Image bytes not provided"
                ))
                return
            }

            // 비동기 처리
            processingScope.launch {
                val detectionResult = processImageAsync(imageBytes, width, height)

                // 메인 스레드에서 결과 반환
                CoroutineScope(Dispatchers.Main).launch {
                    result.success(detectionResult)
                }
            }

        } catch (e: Exception) {
            result.success(mapOf(
                "success" to false,
                "error" to "Detection failed: ${e.message}"
            ))
        }
    }

    private suspend fun processImageAsync(
        imageBytes: ByteArray,
        width: Int,
        height: Int
    ): Map<String, Any> = withContext(Dispatchers.Default) {

        val inferenceTime = measureTimeMillis {
            try {
                // 바이트 배열을 Bitmap으로 변환
                val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                    ?: return@withContext mapOf(
                        "success" to false,
                        "error" to "Failed to decode image"
                    )

                // 크기 조정
                val resizedBitmap = if (bitmap.width != width || bitmap.height != height) {
                    Bitmap.createScaledBitmap(bitmap, width, height, true)
                } else {
                    bitmap
                }

                var poseResult: PoseResult? = null
                val resultLatch = java.util.concurrent.CountDownLatch(1)

                // 임시 결과 리스너 설정
                pose?.setResultListener { result ->
                    poseResult = result
                    resultLatch.countDown()
                }

                // 포즈 검출 실행
                pose?.send(resizedBitmap)

                // 결과 대기 (최대 100ms)
                resultLatch.await(100, java.util.concurrent.TimeUnit.MILLISECONDS)

                bitmap.recycle()
                if (resizedBitmap != bitmap) {
                    resizedBitmap.recycle()
                }

                return@withContext if (poseResult != null) {
                    val landmarks = convertLandmarks(poseResult!!.landmarks())
                    mapOf(
                        "success" to true,
                        "landmarks" to landmarks
                    )
                } else {
                    mapOf(
                        "success" to false,
                        "error" to "No pose detected"
                    )
                }

            } catch (e: MediaPipeException) {
                return@withContext mapOf(
                    "success" to false,
                    "error" to "MediaPipe exception: ${e.message}"
                )
            } catch (e: Exception) {
                return@withContext mapOf(
                    "success" to false,
                    "error" to "Processing exception: ${e.message}"
                )
            }
        }

        // 성능 메트릭 업데이트
        updatePerformanceMetrics(inferenceTime)

        return@withContext mapOf(
            "success" to false,
            "error" to "Unexpected processing end"
        )
    }

    private fun convertLandmarks(landmarks: List<PoseLandmark>): List<Map<String, Any>> {
        return landmarks.mapIndexed { index, landmark ->
            mapOf(
                "id" to index,
                "x" to landmark.x(),
                "y" to landmark.y(),
                "z" to landmark.z(),
                "visibility" to landmark.visibility()
            )
        }
    }

    private fun updatePerformanceMetrics(inferenceTime: Long) {
        frameCount++

        // 최근 100개 프레임의 추론 시간 유지
        inferenceTimeHistory.add(inferenceTime)
        if (inferenceTimeHistory.size > 100) {
            inferenceTimeHistory.removeAt(0)
        }
    }

    private fun getPerformanceMetrics(result: Result) {
        try {
            val currentTime = System.currentTimeMillis()
            val elapsedTime = (currentTime - startTime) / 1000.0
            val fps = if (elapsedTime > 0) frameCount / elapsedTime else 0.0

            val avgInferenceTime = if (inferenceTimeHistory.isNotEmpty()) {
                inferenceTimeHistory.average()
            } else 0.0

            // 시스템 리소스 사용량 (간단한 추정)
            val runtime = Runtime.getRuntime()
            val maxMemory = runtime.maxMemory()
            val usedMemory = runtime.totalMemory() - runtime.freeMemory()
            val memoryUsage = ((usedMemory.toDouble() / maxMemory) * 100).toInt()

            result.success(mapOf(
                "averageInferenceTime" to avgInferenceTime,
                "framesPerSecond" to fps,
                "memoryUsage" to usedMemory,
                "cpuUsage" to getCpuUsage(),
                "gpuUsage" to 0.0 // GPU 사용량은 복잡한 계산 필요
            ))
        } catch (e: Exception) {
            result.error("METRICS_ERROR", "Failed to get performance metrics", e.message)
        }
    }

    private fun getCpuUsage(): Double {
        return try {
            val process = Runtime.getRuntime().exec("top -n 1")
            // 실제 구현에서는 더 정확한 CPU 사용량 계산 필요
            0.0
        } catch (e: Exception) {
            0.0
        }
    }

    private fun dispose(result: Result) {
        try {
            pose?.close()
            pose = null
            isInitialized = false

            // 코루틴 정리
            processingScope.cancel()

            // 성능 메트릭 초기화
            inferenceTimeHistory.clear()
            frameCount = 0
            startTime = System.currentTimeMillis()

            result.success(mapOf("success" to true))
        } catch (e: Exception) {
            result.error("DISPOSE_ERROR", "Failed to dispose MediaPipe", e.message)
        }
    }

    override fun onDetachedFromEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)

        // 리소스 정리
        pose?.close()
        processingScope.cancel()
    }
}

/**
 * 최적화된 이미지 처리 유틸리티
 */
object ImageUtils {

    fun convertYuv420ToRgb(yuv420: ByteArray, width: Int, height: Int): ByteArray {
        val rgbBytes = ByteArray(width * height * 3)

        // YUV420을 RGB로 변환하는 최적화된 알고리즘
        val ySize = width * height
        val uvSize = ySize / 4

        for (i in 0 until height) {
            for (j in 0 until width) {
                val yIndex = i * width + j
                val uvIndex = ySize + (i / 2) * (width / 2) + (j / 2)

                val y = yuv420[yIndex].toInt() and 0xFF
                val u = yuv420[uvIndex].toInt() and 0xFF
                val v = yuv420[uvIndex + uvSize].toInt() and 0xFF

                // YUV to RGB 변환
                val r = (y + 1.402 * (v - 128)).toInt().coerceIn(0, 255)
                val g = (y - 0.344136 * (u - 128) - 0.714136 * (v - 128)).toInt().coerceIn(0, 255)
                val b = (y + 1.772 * (u - 128)).toInt().coerceIn(0, 255)

                val rgbIndex = yIndex * 3
                rgbBytes[rgbIndex] = r.toByte()
                rgbBytes[rgbIndex + 1] = g.toByte()
                rgbBytes[rgbIndex + 2] = b.toByte()
            }
        }

        return rgbBytes
    }

    fun resizeBilinear(
        input: ByteArray,
        inputWidth: Int,
        inputHeight: Int,
        outputWidth: Int,
        outputHeight: Int
    ): ByteArray {
        val output = ByteArray(outputWidth * outputHeight * 3)

        val xRatio = inputWidth.toFloat() / outputWidth
        val yRatio = inputHeight.toFloat() / outputHeight

        for (y in 0 until outputHeight) {
            for (x in 0 until outputWidth) {
                val srcX = (x * xRatio).toInt()
                val srcY = (y * yRatio).toInt()

                val srcIndex = (srcY * inputWidth + srcX) * 3
                val dstIndex = (y * outputWidth + x) * 3

                if (srcIndex < input.size - 2 && dstIndex < output.size - 2) {
                    output[dstIndex] = input[srcIndex]
                    output[dstIndex + 1] = input[srcIndex + 1]
                    output[dstIndex + 2] = input[srcIndex + 2]
                }
            }
        }

        return output
    }
}