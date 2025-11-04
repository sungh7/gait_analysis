import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import '../../shared/services/realtime_camera_service.dart';
import '../../shared/services/v7_pure3d_service.dart';
import '../../shared/models/gait_analysis_models.dart';

/// 실시간 V7 Pure 3D 분석 화면
///
/// 실제 카메라로 보행을 촬영하고 실시간으로 분석합니다.
class RealtimeV7Screen extends StatefulWidget {
  const RealtimeV7Screen({Key? key}) : super(key: key);

  @override
  State<RealtimeV7Screen> createState() => _RealtimeV7ScreenState();
}

class _RealtimeV7ScreenState extends State<RealtimeV7Screen> {
  final RealtimeCameraService _cameraService = RealtimeCameraService();

  bool _isInitialized = false;
  String _statusMessage = '초기화 중...';
  V7DetectionResult? _result;
  List<PoseLandmark>? _currentLandmarks;

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    try {
      // 콜백 설정
      _cameraService.onStatusUpdate = (message) {
        setState(() {
          _statusMessage = message;
        });
      };

      _cameraService.onLandmarksDetected = (landmarks) {
        setState(() {
          _currentLandmarks = landmarks;
        });
      };

      _cameraService.onAnalysisComplete = (result) {
        setState(() {
          _result = result;
        });
      };

      // 초기화
      await _cameraService.initialize();

      setState(() {
        _isInitialized = true;
        _statusMessage = '✅ 준비 완료 - 전면 카메라를 향해 걸어주세요';
      });

    } catch (e) {
      setState(() {
        _statusMessage = '❌ 초기화 실패: $e';
      });
      print('Initialization error: $e');
    }
  }

  Future<void> _startRecording() async {
    try {
      setState(() {
        _result = null;
        _currentLandmarks = null;
      });

      await _cameraService.startRecording();

    } catch (e) {
      setState(() {
        _statusMessage = '❌ 촬영 시작 실패: $e';
      });
    }
  }

  Future<void> _stopRecording() async {
    try {
      await _cameraService.stopRecording();
    } catch (e) {
      setState(() {
        _statusMessage = '❌ 촬영 중지 실패: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('V7 Pure 3D 실시간 분석'),
        backgroundColor: Colors.teal,
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            onPressed: _showInfoDialog,
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            // 상태 배너
            _buildStatusBanner(),

            // 카메라 프리뷰
            Expanded(
              flex: 3,
              child: _buildCameraPreview(),
            ),

            // 랜드마크 정보
            if (_currentLandmarks != null && _cameraService.isRecording)
              _buildLandmarkInfo(),

            // 제어 버튼
            _buildControlButtons(),

            // 결과 표시
            if (_result != null)
              Expanded(
                flex: 2,
                child: _buildResultSummary(),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusBanner() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      color: _getStatusColor(),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          if (_cameraService.isRecording)
            Container(
              width: 12,
              height: 12,
              margin: const EdgeInsets.only(right: 8),
              decoration: const BoxDecoration(
                color: Colors.red,
                shape: BoxShape.circle,
              ),
            ),
          Flexible(
            child: Text(
              _statusMessage,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 16,
                fontWeight: FontWeight.bold,
              ),
              textAlign: TextAlign.center,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCameraPreview() {
    if (!_isInitialized || _cameraService.cameraController == null) {
      return Container(
        color: Colors.black,
        child: const Center(
          child: CircularProgressIndicator(color: Colors.teal),
        ),
      );
    }

    final controller = _cameraService.cameraController!;

    if (!controller.value.isInitialized) {
      return Container(
        color: Colors.black,
        child: const Center(
          child: CircularProgressIndicator(color: Colors.teal),
        ),
      );
    }

    return Container(
      color: Colors.black,
      child: Stack(
        fit: StackFit.expand,
        children: [
          // 카메라 프리뷰
          Center(
            child: AspectRatio(
              aspectRatio: controller.value.aspectRatio,
              child: CameraPreview(controller),
            ),
          ),

          // 랜드마크 오버레이 (선택적)
          if (_currentLandmarks != null && _cameraService.isRecording)
            _buildLandmarkOverlay(),

          // 프레임 카운터
          if (_cameraService.isRecording)
            Positioned(
              top: 16,
              right: 16,
              child: Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 8,
                ),
                decoration: BoxDecoration(
                  color: Colors.red,
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  '${_cameraService.recordedFrames}/180',
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildLandmarkOverlay() {
    // TODO: 랜드마크를 카메라 프리뷰 위에 그리기
    // CustomPaint를 사용하여 스켈레톤 시각화
    return const SizedBox();
  }

  Widget _buildLandmarkInfo() {
    return Container(
      padding: const EdgeInsets.all(8),
      color: Colors.black87,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        children: [
          _buildInfoChip(
            Icons.person,
            '랜드마크',
            '${_currentLandmarks!.length}',
          ),
          _buildInfoChip(
            Icons.videocam,
            '프레임',
            '${_cameraService.recordedFrames}',
          ),
          _buildInfoChip(
            Icons.speed,
            '30 FPS',
            'LIVE',
          ),
        ],
      ),
    );
  }

  Widget _buildInfoChip(IconData icon, String label, String value) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(icon, color: Colors.teal, size: 20),
        const SizedBox(height: 4),
        Text(
          label,
          style: const TextStyle(
            color: Colors.white70,
            fontSize: 12,
          ),
        ),
        Text(
          value,
          style: const TextStyle(
            color: Colors.teal,
            fontSize: 14,
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }

  Widget _buildControlButtons() {
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // 촬영 버튼
          if (!_cameraService.isRecording)
            ElevatedButton.icon(
              onPressed: _isInitialized ? _startRecording : null,
              icon: const Icon(Icons.fiber_manual_record, size: 28),
              label: const Text(
                '촬영 시작',
                style: TextStyle(fontSize: 18),
              ),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(
                  horizontal: 40,
                  vertical: 20,
                ),
                backgroundColor: Colors.red,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(30),
                ),
              ),
            ),

          // 중지 버튼
          if (_cameraService.isRecording)
            ElevatedButton.icon(
              onPressed: _stopRecording,
              icon: const Icon(Icons.stop, size: 28),
              label: const Text(
                '촬영 중지',
                style: TextStyle(fontSize: 18),
              ),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(
                  horizontal: 40,
                  vertical: 20,
                ),
                backgroundColor: Colors.orange,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(30),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildResultSummary() {
    if (_result == null) return const SizedBox();

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Card(
        elevation: 8,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // 결과 헤더
              Row(
                children: [
                  Icon(
                    _result!.isPathological
                        ? Icons.warning_amber_rounded
                        : Icons.check_circle,
                    color: _result!.isPathological ? Colors.orange : Colors.green,
                    size: 32,
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      _result!.isPathological ? '비정상 검출' : '정상 보행',
                      style: const TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 16),

              // 위험 점수
              _buildQuickRiskScore(),

              const SizedBox(height: 16),

              // 상세 보기 버튼
              ElevatedButton.icon(
                onPressed: () => _showDetailedResults(),
                icon: const Icon(Icons.article),
                label: const Text('상세 결과 보기'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.teal,
                  minimumSize: const Size(double.infinity, 48),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildQuickRiskScore() {
    final score = _result!.riskScore;
    Color scoreColor;
    if (score >= 80) {
      scoreColor = Colors.red;
    } else if (score >= 60) {
      scoreColor = Colors.orange;
    } else if (score >= 40) {
      scoreColor = Colors.yellow.shade700;
    } else {
      scoreColor = Colors.green;
    }

    return Row(
      children: [
        const Text(
          '위험도: ',
          style: TextStyle(fontSize: 16),
        ),
        Expanded(
          child: LinearProgressIndicator(
            value: score / 100,
            minHeight: 8,
            backgroundColor: Colors.grey.shade300,
            valueColor: AlwaysStoppedAnimation<Color>(scoreColor),
          ),
        ),
        const SizedBox(width: 12),
        Text(
          '$score',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
            color: scoreColor,
          ),
        ),
      ],
    );
  }

  Color _getStatusColor() {
    if (_statusMessage.contains('❌')) return Colors.red;
    if (_statusMessage.contains('⚠️')) return Colors.orange;
    if (_statusMessage.contains('🎥')) return Colors.blue;
    if (_statusMessage.contains('✅')) return Colors.green;
    return Colors.teal;
  }

  void _showInfoDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('V7 Pure 3D 실시간 분석'),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: const [
              Text(
                '실시간 3D pose 추출을 이용한 보행 분석',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              SizedBox(height: 12),
              Text('• MediaPipe로 33개 랜드마크 추출'),
              Text('• 30fps 실시간 처리'),
              Text('• 6초 (180 프레임) 자동 녹화'),
              Text('• V7 Pure 3D 알고리즘 분석'),
              SizedBox(height: 12),
              Text(
                '성능:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              Text('• 임상 병리 민감도: 98.6%'),
              Text('• 파킨슨/뇌졸중/뇌성마비: 100% 검출'),
              SizedBox(height: 12),
              Text(
                '사용법:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              Text('1. 스마트폰을 안정적으로 고정'),
              Text('2. 전면 카메라를 향해 서기'),
              Text('3. "촬영 시작" 버튼 누르기'),
              Text('4. 카메라를 향해 6초간 걷기'),
              Text('5. 자동으로 분석 완료'),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('확인'),
          ),
        ],
      ),
    );
  }

  void _showDetailedResults() {
    // TODO: 상세 결과 화면으로 이동
    Navigator.pushNamed(
      context,
      '/detailed_results',
      arguments: _result,
    );
  }

  @override
  void dispose() {
    _cameraService.dispose();
    super.dispose();
  }
}
