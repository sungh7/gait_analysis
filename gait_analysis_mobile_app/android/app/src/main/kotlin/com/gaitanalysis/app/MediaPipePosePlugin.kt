package com.gaitanalysis.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.annotation.NonNull
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result
import kotlinx.coroutines.*
import java.io.ByteArrayInputStream

/**
 * MediaPipe Pose Detection Plugin for Flutter
 *
 * V7 Pure 3D 알고리즘을 위한 실시간 3D pose 추출
 * - 33개 랜드마크 (MediaPipe Pose)
 * - 3D 좌표 (x, y, z) + visibility
 * - 실시간 비디오 처리
 */
class MediaPipePosePlugin : FlutterPlugin, MethodCallHandler {
    private lateinit var channel: MethodChannel
    private lateinit var context: Context

    private var poseLandmarker: PoseLandmarker? = null
    private val coroutineScope = CoroutineScope(Dispatchers.Main + Job())

    companion object {
        private const val CHANNEL_NAME = "com.gaitanalysis.app/mediapipe_pose"
        private const val MODEL_ASSET_PATH = "pose_landmarker_lite.task"

        // MediaPipe Pose landmark indices
        const val LEFT_HEEL = 29
        const val RIGHT_HEEL = 30
        const val LEFT_HIP = 23
        const val RIGHT_HIP = 24
        const val LEFT_SHOULDER = 11
        const val RIGHT_SHOULDER = 12
        const val LEFT_ANKLE = 27
        const val RIGHT_ANKLE = 28
    }

    override fun onAttachedToEngine(@NonNull flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(flutterPluginBinding.binaryMessenger, CHANNEL_NAME)
        channel.setMethodCallHandler(this)
        context = flutterPluginBinding.applicationContext
    }

    override fun onMethodCall(@NonNull call: MethodCall, @NonNull result: Result) {
        when (call.method) {
            "initialize" -> {
                initializePoseLandmarker(result)
            }
            "detectPose" -> {
                val imageBytes = call.argument<ByteArray>("imageBytes")
                val timestampMs = call.argument<Long>("timestampMs") ?: System.currentTimeMillis()

                if (imageBytes == null) {
                    result.error("INVALID_ARGUMENT", "imageBytes is required", null)
                    return
                }

                detectPoseFromBytes(imageBytes, timestampMs, result)
            }
            "detectPoseVideo" -> {
                val imageBytes = call.argument<ByteArray>("imageBytes")
                val timestampMs = call.argument<Long>("timestampMs") ?: System.currentTimeMillis()

                if (imageBytes == null) {
                    result.error("INVALID_ARGUMENT", "imageBytes is required", null)
                    return
                }

                detectPoseVideoMode(imageBytes, timestampMs, result)
            }
            "dispose" -> {
                disposePoseLandmarker(result)
            }
            else -> {
                result.notImplemented()
            }
        }
    }

    private fun initializePoseLandmarker(result: Result) {
        try {
            val baseOptionsBuilder = BaseOptions.builder()
                .setModelAssetPath(MODEL_ASSET_PATH)

            val options = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(baseOptionsBuilder.build())
                .setRunningMode(RunningMode.VIDEO)
                .setNumPoses(1)
                .setMinPoseDetectionConfidence(0.5f)
                .setMinPosePresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .setOutputSegmentationMasks(false)
                .build()

            poseLandmarker = PoseLandmarker.createFromOptions(context, options)

            result.success(mapOf(
                "status" to "initialized",
                "modelPath" to MODEL_ASSET_PATH,
                "runningMode" to "VIDEO",
                "numPoses" to 1
            ))

        } catch (e: Exception) {
            result.error(
                "INITIALIZATION_ERROR",
                "Failed to initialize PoseLandmarker: ${e.message}",
                e.stackTraceToString()
            )
        }
    }

    private fun detectPoseFromBytes(
        imageBytes: ByteArray,
        timestampMs: Long,
        result: Result
    ) {
        coroutineScope.launch(Dispatchers.Default) {
            try {
                val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                    ?: throw IllegalArgumentException("Failed to decode image bytes")

                val mpImage = BitmapImageBuilder(bitmap).build()

                val detectionResult = poseLandmarker?.detectForVideo(mpImage, timestampMs)
                    ?: throw IllegalStateException("PoseLandmarker not initialized")

                val landmarks = extractLandmarks(detectionResult)

                withContext(Dispatchers.Main) {
                    result.success(mapOf(
                        "landmarks" to landmarks,
                        "timestampMs" to timestampMs,
                        "numPoses" to detectionResult.landmarks().size
                    ))
                }

            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    result.error(
                        "DETECTION_ERROR",
                        "Failed to detect pose: ${e.message}",
                        e.stackTraceToString()
                    )
                }
            }
        }
    }

    private fun detectPoseVideoMode(
        imageBytes: ByteArray,
        timestampMs: Long,
        result: Result
    ) {
        // Video mode - optimized for streaming
        detectPoseFromBytes(imageBytes, timestampMs, result)
    }

    private fun extractLandmarks(result: PoseLandmarkerResult): List<Map<String, Any>> {
        val landmarks = mutableListOf<Map<String, Any>>()

        if (result.landmarks().isEmpty()) {
            return landmarks
        }

        // Get first pose (we only detect 1 person)
        val poseLandmarks = result.landmarks()[0]
        val worldLandmarks = result.worldLandmarks()[0]

        for (i in poseLandmarks.indices) {
            val landmark = poseLandmarks[i]
            val worldLandmark = worldLandmarks[i]

            landmarks.add(mapOf(
                "id" to i,
                "position" to getLandmarkName(i),
                // Normalized coordinates (0-1)
                "x" to landmark.x(),
                "y" to landmark.y(),
                "z" to landmark.z(),
                "visibility" to landmark.visibility().orElse(1.0f),
                // World coordinates (meters)
                "worldX" to worldLandmark.x(),
                "worldY" to worldLandmark.y(),
                "worldZ" to worldLandmark.z()
            ))
        }

        return landmarks
    }

    private fun getLandmarkName(index: Int): String {
        return when (index) {
            0 -> "nose"
            1 -> "left_eye_inner"
            2 -> "left_eye"
            3 -> "left_eye_outer"
            4 -> "right_eye_inner"
            5 -> "right_eye"
            6 -> "right_eye_outer"
            7 -> "left_ear"
            8 -> "right_ear"
            9 -> "mouth_left"
            10 -> "mouth_right"
            11 -> "left_shoulder"
            12 -> "right_shoulder"
            13 -> "left_elbow"
            14 -> "right_elbow"
            15 -> "left_wrist"
            16 -> "right_wrist"
            17 -> "left_pinky"
            18 -> "right_pinky"
            19 -> "left_index"
            20 -> "right_index"
            21 -> "left_thumb"
            22 -> "right_thumb"
            23 -> "left_hip"
            24 -> "right_hip"
            25 -> "left_knee"
            26 -> "right_knee"
            27 -> "left_ankle"
            28 -> "right_ankle"
            29 -> "left_heel"
            30 -> "right_heel"
            31 -> "left_foot_index"
            32 -> "right_foot_index"
            else -> "unknown"
        }
    }

    private fun disposePoseLandmarker(result: Result) {
        try {
            poseLandmarker?.close()
            poseLandmarker = null
            coroutineScope.cancel()

            result.success(mapOf("status" to "disposed"))
        } catch (e: Exception) {
            result.error(
                "DISPOSAL_ERROR",
                "Failed to dispose PoseLandmarker: ${e.message}",
                e.stackTraceToString()
            )
        }
    }

    override fun onDetachedFromEngine(@NonNull binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
        poseLandmarker?.close()
        poseLandmarker = null
        coroutineScope.cancel()
    }
}
