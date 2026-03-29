// ============================================================
//  VISION ASSISTANT — Flutter App
//  Backend: YOLOv8 + EasyOCR + Ollama + gTTS over WebSocket
//
//  FIXED: Added voice input (speech_to_text) + text command input
//         so Flutter can send voice_command messages to Python backend.
//
//  pubspec.yaml dependencies (add these):
//    web_socket_channel: ^2.4.0
//    flutter_tts: ^3.8.5
//    camera: ^0.10.5
//    speech_to_text: ^6.6.0     ← NEW
//
//  AndroidManifest.xml (inside <manifest>):
//    <uses-permission android:name="android.permission.INTERNET"/>
//    <uses-permission android:name="android.permission.CAMERA"/>
//    <uses-permission android:name="android.permission.RECORD_AUDIO"/>   ← NEW
//
//  iOS Info.plist (add):
//    <key>NSMicrophoneUsageDescription</key>
//    <string>Microphone is used for voice commands</string>
//    <key>NSSpeechRecognitionUsageDescription</key>
//    <string>Speech recognition is used for voice commands</string>
//
//  Set SERVER_IP to your laptop's WiFi IP address.
// ============================================================

import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:camera/camera.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

// ── CHANGE THIS ────────────────────────────────────────────
const String SERVER_IP   = '192.168.172.93';
const int    SERVER_PORT = 8765;
// ───────────────────────────────────────────────────────────

const Map<String, String> kSourceLanguages = {
  'en-US': 'English',
  'hi-IN': 'Hindi',
};

const Map<String, String> kTargetLanguages = {
  'en': 'English',
  'hi': 'Hindi',
};

const Map<String, String> kTtsLocale = {
  'en': 'en-US',
  'hi': 'hi-IN',
};

// speech_to_text locale IDs per source language
const Map<String, String> kSttLocale = {
  'en-US': 'en_US',
  'hi-IN': 'hi_IN',
};

// ─────────────────────────────────────────────────────────────
//  MODELS
// ─────────────────────────────────────────────────────────────
class AlertEntry {
  final String   text;
  final DateTime ts;
  final bool     isCritical;
  AlertEntry({required this.text, required this.ts, required this.isCritical});
}

// ─────────────────────────────────────────────────────────────
//  APP
// ─────────────────────────────────────────────────────────────
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.immersiveSticky);
  runApp(const VisionApp());
}

class VisionApp extends StatelessWidget {
  const VisionApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
        debugShowCheckedModeBanner: false,
        theme: ThemeData.dark().copyWith(
          scaffoldBackgroundColor: Colors.black,
        ),
        home: const VisionScreen(),
      );
}

class _BoundingBox extends StatelessWidget {
  final String label;
  final double confidence;
  final bool   critical;

  const _BoundingBox({
    required this.label,
    required this.confidence,
    required this.critical,
  });

  @override
  Widget build(BuildContext context) {
    final color = critical ? Colors.redAccent : kCyan;
    return Container(
      decoration: BoxDecoration(border: Border.all(color: color, width: 2)),
      child: Align(
        alignment: Alignment.topLeft,
        child: Container(
          color:   color,
          padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
          child:   Text(
            '${label.toUpperCase()} ${(confidence * 100).toInt()}%',
            style: const TextStyle(fontSize: 10, color: Colors.white),
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────
//  MAIN SCREEN
// ─────────────────────────────────────────────────────────────
class VisionScreen extends StatefulWidget {
  const VisionScreen({super.key});

  @override
  State<VisionScreen> createState() => _VisionScreenState();
}

class _VisionScreenState extends State<VisionScreen>
    with TickerProviderStateMixin {

  // ── WebSocket ──────────────────────────────────────────────
  WebSocketChannel? _channel;
  bool   _connected  = false;
  bool   _connecting = false;
  String _statusMsg  = 'Tap Connect to start';

  // ── Feed ──────────────────────────────────────────────────
  Uint8List?     _frameBytes;
  List<dynamic>  _detections = [];

  // ── Alerts ─────────────────────────────────────────────────
  final List<AlertEntry> _alerts = [];
  AlertEntry? _banner;
  Timer?      _bannerTimer;

  // ── Priorities ─────────────────────────────────────────────
  List<String> _priorities    = [];
  List<String> _criticalList  = [];

  // ── Language ───────────────────────────────────────────────
  String _sourceLanguage = 'en-US';
  String _targetLanguage = 'en';

  // ── TTS ────────────────────────────────────────────────────
  late FlutterTts _tts;
  bool _ttsEnabled = true;
  bool _ttsReady   = false;

  // ── Camera ─────────────────────────────────────────────────
  CameraController? _cameraController;
  Timer?            _captureTimer;

  // ── Voice (speech_to_text) ─────────────────────────────────
  final stt.SpeechToText _speech    = stt.SpeechToText();
  bool   _sttAvailable  = false;
  bool   _isListening   = false;
  String _liveText      = '';        // live partial transcript shown in mic bubble
  String _lastCommand   = '';        // last sent command (shown in panel)

  // ── Text command input ─────────────────────────────────────
  final TextEditingController _cmdCtrl = TextEditingController();
  bool _showCmdInput = false;

  // ── UI ─────────────────────────────────────────────────────
  bool _showPanel = false;
  int  _panelTab  = 0;    // 0=Alerts  1=Priorities  2=Voice  3=Settings

  // ── Animations ─────────────────────────────────────────────
  late AnimationController _pulseCtrl;
  late Animation<double>   _pulseAnim;
  late AnimationController _panelCtrl;
  late Animation<Offset>   _panelSlide;
  late AnimationController _bannerCtrl;
  late Animation<double>   _bannerAnim;
  late AnimationController _micPulseCtrl;
  late Animation<double>   _micPulseAnim;

  // ─────────────────────────────────────────────────────────
  @override
  void initState() {
    super.initState();
    _initTts();
    _initCamera();
    _initStt();

    _pulseCtrl = AnimationController(vsync: this, duration: const Duration(seconds: 2))
      ..repeat(reverse: true);
    _pulseAnim = Tween<double>(begin: 0.4, end: 1.0)
        .animate(CurvedAnimation(parent: _pulseCtrl, curve: Curves.easeInOut));

    _panelCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 300));
    _panelSlide = Tween<Offset>(begin: const Offset(1, 0), end: Offset.zero)
        .animate(CurvedAnimation(parent: _panelCtrl, curve: Curves.easeOut));

    _bannerCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 450));
    _bannerAnim = CurvedAnimation(parent: _bannerCtrl, curve: Curves.elasticOut);

    _micPulseCtrl = AnimationController(vsync: this, duration: const Duration(milliseconds: 900))
      ..repeat(reverse: true);
    _micPulseAnim = Tween<double>(begin: 1.0, end: 1.25)
        .animate(CurvedAnimation(parent: _micPulseCtrl, curve: Curves.easeInOut));
  }

  // ─────────────────────────────────────────
  //  Init helpers
  // ─────────────────────────────────────────
  Future<void> _initTts() async {
    _tts = FlutterTts();
    await _tts.setLanguage(kTtsLocale[_targetLanguage] ?? 'en-US');
    await _tts.setSpeechRate(0.48);
    await _tts.setVolume(1.0);
    await _tts.setPitch(1.05);
    if (mounted) setState(() => _ttsReady = true);
  }

  Future<void> _updateTtsLocale(String targetLanguage) async {
    final locale = kTtsLocale[targetLanguage] ?? 'en-US';
    await _tts.setLanguage(locale);
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    if (cameras.isNotEmpty) {
      _cameraController = CameraController(cameras[0], ResolutionPreset.medium);
      await _cameraController!.initialize();
      if (mounted) setState(() {});
    }
  }

  Future<void> _initStt() async {
    final available = await _speech.initialize(
      onError: (e) {
        debugPrint('[STT Error] $e');
        if (mounted) setState(() => _isListening = false);
      },
      onStatus: (status) {
        debugPrint('[STT Status] $status');
        // When STT engine says "done" or "notListening", finalize
        if (status == 'done' || status == 'notListening') {
          if (_isListening) _stopListening(send: true);
        }
      },
    );
    if (mounted) setState(() => _sttAvailable = available);
  }

  @override
  void dispose() {
    _channel?.sink.close();
    _cameraController?.dispose();
    _captureTimer?.cancel();
    _pulseCtrl.dispose();
    _panelCtrl.dispose();
    _bannerCtrl.dispose();
    _micPulseCtrl.dispose();
    _bannerTimer?.cancel();
    _cmdCtrl.dispose();
    _tts.stop();
    _speech.stop();
    super.dispose();
  }

  // ─────────────────────────────────────────
  //  WebSocket lifecycle
  // ─────────────────────────────────────────
  void _connect() {
    if (_connecting || _connected) return;
    setState(() {
      _connecting = true;
      _statusMsg  = 'Connecting…';
    });
    try {
      _channel = WebSocketChannel.connect(Uri.parse('ws://$SERVER_IP:$SERVER_PORT'));
      _channel!.stream.listen(_onMsg, onError: _onError, onDone: _onDone);
      setState(() {
        _connected  = true;
        _connecting = false;
        _statusMsg  = 'Connected  •  $SERVER_IP';
      });
      _startCapture();
    } catch (e) {
      setState(() {
        _connecting = false;
        _statusMsg  = 'Failed: $e';
      });
    }
  }

  void _disconnect() {
    _channel?.sink.close();
    _captureTimer?.cancel();
    _captureTimer = null;
    setState(() {
      _connected   = false;
      _frameBytes  = null;
      _detections  = [];
      _statusMsg   = 'Disconnected';
    });
  }

  // ─────────────────────────────────────────
  //  Camera → backend frame streaming
  // ─────────────────────────────────────────
  void _startCapture() {
    _captureTimer = Timer.periodic(
      const Duration(milliseconds: 1000 ~/ 15),
      _captureAndSend,
    );
  }

  void _captureAndSend(Timer t) async {
    if (_cameraController == null ||
        !_cameraController!.value.isInitialized ||
        !_connected) return;
    try {
      final image = await _cameraController!.takePicture();
      final bytes = await image.readAsBytes();
      _channel!.sink.add(jsonEncode({
        'type': 'frame',
        'data': base64Encode(bytes),
      }));
    } catch (e) {
      debugPrint('Capture error: $e');
    }
  }

  // ─────────────────────────────────────────
  //  Voice command — speech_to_text
  // ─────────────────────────────────────────

  /// Start listening. Automatically stops after [listenFor] duration.
  void _startListening() {
    if (!_sttAvailable || _isListening) return;
    final localeId = kSttLocale[_sourceLanguage] ?? 'en_US';

    setState(() {
      _isListening = true;
      _liveText    = '';
    });

    _speech.listen(
      localeId:   localeId,
      listenFor:  const Duration(seconds: 10),
      pauseFor:   const Duration(seconds: 2),
      onResult:   (result) {
        setState(() => _liveText = result.recognizedWords);
        if (result.finalResult) {
          _stopListening(send: true);
        }
      },
    );
  }

  /// Stop listening and optionally send the recognized text to the backend.
  void _stopListening({bool send = false}) {
    if (!_isListening) return;
    _speech.stop();
    final text = _liveText.trim();
    setState(() {
      _isListening = false;
      _liveText    = '';
    });
    if (send && text.isNotEmpty) {
      _sendVoiceCommand(text);
    }
  }

  /// Send a text string to backend as a voice_command WebSocket message.
  void _sendVoiceCommand(String text) {
    if (!_connected || text.isEmpty) return;
    _channel!.sink.add(jsonEncode({
      'type': 'voice_command',
      'text': text,
    }));
    setState(() => _lastCommand = text);
    debugPrint('[Voice→WS] Sent: "$text"');
    // Show a snackbar confirmation
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Row(children: [
            const Icon(Icons.send_rounded, color: kCyan, size: 14),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                'Sent: "$text"',
                style: const TextStyle(color: Colors.white, fontSize: 12),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ]),
          backgroundColor: const Color(0xFF0D1428),
          behavior:        SnackBarBehavior.floating,
          duration:        const Duration(seconds: 2),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(10),
            side:         const BorderSide(color: kCyan, width: 1),
          ),
        ),
      );
    }
  }

  // ─────────────────────────────────────────
  //  Language — send to backend
  // ─────────────────────────────────────────
  void _sendLanguage({String? source, String? target}) {
    final newSource = source ?? _sourceLanguage;
    final newTarget = target ?? _targetLanguage;

    if (!_connected) {
      setState(() {
        _sourceLanguage = newSource;
        _targetLanguage = newTarget;
      });
      _updateTtsLocale(newTarget);
      return;
    }
    _channel!.sink.add(jsonEncode({
      'type':            'set_language',
      'source_language': newSource,
      'target_language': newTarget,
    }));
    setState(() {
      _sourceLanguage = newSource;
      _targetLanguage = newTarget;
    });
    _updateTtsLocale(newTarget);
  }

  // ─────────────────────────────────────────
  //  WebSocket message handler
  // ─────────────────────────────────────────
  void _onMsg(dynamic raw) {
    try {
      final msg  = jsonDecode(raw as String) as Map<String, dynamic>;
      final type = msg['type'] as String;

      switch (type) {
        case 'frame':
          final dataStr  = msg['data'] as String?;
          final detects  = (msg['detections'] as List<dynamic>? ?? []);
          setState(() {
            _detections = detects;
            if (dataStr != null) _frameBytes = base64Decode(dataStr);
          });

        case 'alert':
          _handleAlert(msg['text'] as String);

        case 'init':
          final src = msg['source_language'] as String?;
          final tgt = msg['target_language'] as String?;
          setState(() {
            _priorities   = List<String>.from(msg['priorities'] as List? ?? []);
            _criticalList = List<String>.from(msg['critical']   as List? ?? []);
            if (src != null && kSourceLanguages.containsKey(src)) _sourceLanguage = src;
            if (tgt != null && kTargetLanguages.containsKey(tgt)) _targetLanguage = tgt;
          });
          if (tgt != null) _updateTtsLocale(tgt);

        case 'priorities_updated':
          setState(() {
            _priorities = List<String>.from(msg['priorities'] as List? ?? []);
          });

        case 'language_updated':
          final src = msg['source_language'] as String?;
          final tgt = msg['target_language'] as String?;
          setState(() {
            if (src != null && kSourceLanguages.containsKey(src)) _sourceLanguage = src;
            if (tgt != null && kTargetLanguages.containsKey(tgt)) _targetLanguage = tgt;
          });
          if (tgt != null) _updateTtsLocale(tgt);
      }
    } catch (e) {
      debugPrint('[WS parse] $e');
    }
  }

  void _onError(Object e) => setState(() {
        _connected = false;
        _statusMsg = 'Error: $e';
      });

  void _onDone() => setState(() {
        _connected = false;
        _statusMsg = 'Connection closed — tap Connect to retry';
      });

  // ─────────────────────────────────────────
  //  Alert + TTS
  // ─────────────────────────────────────────
  bool _isCritical(String t) {
    final lo = t.toLowerCase();
    if (_criticalList.any((w) => lo.contains(w))) return true;
    const fallback = ['danger', 'fire', 'explosive', 'high voltage', 'poison', 'emergency'];
    return fallback.any((w) => lo.contains(w));
  }

  void _handleAlert(String text) {
    final crit  = _isCritical(text);
    final entry = AlertEntry(text: text, ts: DateTime.now(), isCritical: crit);
    setState(() {
      _banner = entry;
      _alerts.insert(0, entry);
      if (_alerts.length > 60) _alerts.removeLast();
    });
    _bannerCtrl.forward(from: 0);
    _bannerTimer?.cancel();
    _bannerTimer = Timer(
      const Duration(seconds: 7),
      () { if (mounted) setState(() => _banner = null); },
    );
    if (_ttsEnabled && _ttsReady) {
      _tts.stop();
      _tts.speak(text);
    }
    crit ? HapticFeedback.heavyImpact() : HapticFeedback.mediumImpact();
  }

  void _togglePanel() {
    setState(() => _showPanel = !_showPanel);
    _showPanel ? _panelCtrl.forward() : _panelCtrl.reverse();
  }

  // ─────────────────────────────────────────
  //  BUILD
  // ─────────────────────────────────────────
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(children: [
        Positioned.fill(child: _buildFeed()),
        Positioned(top: 0, left: 0, right: 0, child: _buildHud()),

        // Mic bubble (bottom-left, always visible when connected)
        if (_connected)
          Positioned(
            bottom: 28,
            right:  22,
            child: _buildMicButton(),
          ),

        if (_banner != null)
          Positioned(
            bottom: 100, left: 14, right: 80,
            child: _buildBanner(_banner!),
          ),

        Positioned(
          top: 0, bottom: 0, right: 0, width: 300,
          child: SlideTransition(
            position: _panelSlide,
            child:    _buildSidePanel(),
          ),
        ),
      ]),
    );
  }

  // ─────────────────────────────────────────
  //  Mic button (floating, always accessible)
  // ─────────────────────────────────────────
  Widget _buildMicButton() {
    if (!_sttAvailable) {
      return _buildTextCommandButton();
    }

    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        // Live transcript bubble
        if (_isListening && _liveText.isNotEmpty)
          Container(
            constraints: const BoxConstraints(maxWidth: 200),
            margin:      const EdgeInsets.only(bottom: 8),
            padding:     const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color:        Colors.black.withOpacity(0.88),
              borderRadius: BorderRadius.circular(12),
              border:       Border.all(color: kCyan.withOpacity(0.6)),
            ),
            child: Text(
              _liveText,
              style: const TextStyle(color: Colors.white, fontSize: 12),
              maxLines: 3,
              overflow: TextOverflow.ellipsis,
            ),
          ),

        // Keyboard shortcut
        GestureDetector(
          onTap: () => setState(() => _showCmdInput = !_showCmdInput),
          child: Container(
            margin:  const EdgeInsets.only(bottom: 8),
            width:   40, height: 40,
            decoration: BoxDecoration(
              shape:  BoxShape.circle,
              color:  Colors.black54,
              border: Border.all(color: Colors.white24),
            ),
            child: const Icon(Icons.keyboard_rounded,
                color: Colors.white38, size: 18),
          ),
        ),

        // Hold-to-talk mic button
        GestureDetector(
          onLongPressStart: (_) => _startListening(),
          onLongPressEnd:   (_) => _stopListening(send: true),
          onTap:            _isListening
              ? () => _stopListening(send: true)
              : _startListening,
          child: AnimatedBuilder(
            animation: _micPulseAnim,
            builder:   (_, child) => Transform.scale(
              scale: _isListening ? _micPulseAnim.value : 1.0,
              child: child,
            ),
            child: Container(
              width:  62, height: 62,
              decoration: BoxDecoration(
                shape:  BoxShape.circle,
                color:  _isListening
                    ? Colors.redAccent.withOpacity(0.9)
                    : kCyan.withOpacity(0.15),
                border: Border.all(
                  color: _isListening ? Colors.redAccent : kCyan,
                  width: 2,
                ),
                boxShadow: [
                  BoxShadow(
                    color:      (_isListening ? Colors.redAccent : kCyan)
                        .withOpacity(0.45),
                    blurRadius: _isListening ? 20 : 12,
                    spreadRadius: _isListening ? 4 : 1,
                  ),
                ],
              ),
              child: Icon(
                _isListening
                    ? Icons.stop_rounded
                    : Icons.mic_rounded,
                color: _isListening ? Colors.white : kCyan,
                size: 28,
              ),
            ),
          ),
        ),

        // Hint label
        const SizedBox(height: 4),
        Text(
          _isListening ? 'Tap to stop' : 'Hold or tap to speak',
          style: const TextStyle(color: Colors.white38, fontSize: 9),
        ),
      ],
    );
  }

  /// Fallback: small keyboard icon that opens the text input when STT is unavailable
  Widget _buildTextCommandButton() {
    return GestureDetector(
      onTap: () => setState(() => _showCmdInput = !_showCmdInput),
      child: Container(
        width: 52, height: 52,
        decoration: BoxDecoration(
          shape:  BoxShape.circle,
          color:  kCyan.withOpacity(0.12),
          border: Border.all(color: kCyan, width: 1.5),
          boxShadow: [BoxShadow(color: kCyan.withOpacity(0.3), blurRadius: 12)],
        ),
        child: const Icon(Icons.keyboard_voice_rounded, color: kCyan, size: 24),
      ),
    );
  }

  // ─────────────────────────────────────────
  //  Feed
  // ─────────────────────────────────────────
  Widget _buildFeed() {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return _buildNoFeed();
    }

    return LayoutBuilder(builder: (ctx, constraints) {
      final dw = constraints.maxWidth;
      final dh = constraints.maxHeight;
      final previewSize = _cameraController!.value.previewSize!;
      final srcRatio = previewSize.height / previewSize.width;
      final dstRatio = dw / dh;

      double fitW, fitH, ox = 0, oy = 0;
      if (dstRatio > srcRatio) {
        fitH = dh; fitW = dh * srcRatio; ox = (dw - fitW) / 2;
      } else {
        fitW = dw; fitH = dw / srcRatio; oy = (dh - fitH) / 2;
      }

      final sx = fitW / previewSize.height;
      final sy = fitH / previewSize.width;

      return Stack(children: [
        Positioned(left: ox, top: oy, width: fitW, height: fitH,
            child: CameraPreview(_cameraController!)),

        ..._detections.map((d) {
          final box = List<int>.from(d['box']);
          final x1r = 720 - box[3].toDouble();
          final y1r = box[0].toDouble();
          final x2r = 720 - box[1].toDouble();
          final y2r = box[2].toDouble();
          return Positioned(
            left: x1r * sx + ox, top: y1r * sy + oy,
            width: (x2r - x1r) * sx, height: (y2r - y1r) * sy,
            child: _BoundingBox(
              label:      d['label'],
              confidence: (d['confidence'] as num).toDouble(),
              critical:   _isCritical(d['label']),
            ),
          );
        }),

        Positioned(
          left: 14, bottom: 14,
          child: _Chip(
            icon:  Icons.center_focus_strong_rounded,
            label: _detections.isEmpty
                ? 'No detections'
                : '${_detections.length} detected',
            color: _detections.isEmpty ? Colors.white30 : kCyan,
          ),
        ),

        // Floating text command overlay
        if (_showCmdInput && _connected)
          Positioned(
            bottom: 110, left: 14, right: 80,
            child: _buildTextCommandInput(),
          ),
      ]);
    });
  }

  Widget _buildNoFeed() => const Center(
        child: Text('Camera not ready',
            style: TextStyle(color: Colors.white38)),
      );

  // ─────────────────────────────────────────
  //  Text command input overlay
  // ─────────────────────────────────────────
  Widget _buildTextCommandInput() {
    return Container(
      padding:    const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color:        Colors.black.withOpacity(0.92),
        borderRadius: BorderRadius.circular(14),
        border:       Border.all(color: kCyan.withOpacity(0.5)),
        boxShadow:    [BoxShadow(color: kCyan.withOpacity(0.18), blurRadius: 16)],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          const Text(
            'TYPE A COMMAND',
            style: TextStyle(
                color:    kCyan,
                fontSize: 9,
                fontWeight: FontWeight.w700,
                letterSpacing: 1.2),
          ),
          const SizedBox(height: 4),
          Text(
            'e.g. "add pharmacy"  "list"  "remove hospital"',
            style: TextStyle(color: Colors.white30, fontSize: 9),
          ),
          const SizedBox(height: 8),
          Row(children: [
            Expanded(
              child: TextField(
                controller: _cmdCtrl,
                autofocus:  true,
                style: const TextStyle(color: Colors.white, fontSize: 13),
                decoration: InputDecoration(
                  hintText:        'Type command...',
                  hintStyle:       TextStyle(color: Colors.white30, fontSize: 12),
                  contentPadding:  const EdgeInsets.symmetric(
                      horizontal: 10, vertical: 8),
                  filled:          true,
                  fillColor:       Colors.white.withOpacity(0.06),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                    borderSide:  BorderSide(color: kCyan.withOpacity(0.3)),
                  ),
                  enabledBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                    borderSide:  BorderSide(color: kCyan.withOpacity(0.25)),
                  ),
                  focusedBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                    borderSide:  const BorderSide(color: kCyan),
                  ),
                ),
                onSubmitted: (val) {
                  final t = val.trim();
                  if (t.isNotEmpty) {
                    _sendVoiceCommand(t);
                    _cmdCtrl.clear();
                    setState(() => _showCmdInput = false);
                  }
                },
              ),
            ),
            const SizedBox(width: 8),
            GestureDetector(
              onTap: () {
                final t = _cmdCtrl.text.trim();
                if (t.isNotEmpty) {
                  _sendVoiceCommand(t);
                  _cmdCtrl.clear();
                  setState(() => _showCmdInput = false);
                }
              },
              child: Container(
                width: 38, height: 38,
                decoration: BoxDecoration(
                  color:        kCyan.withOpacity(0.18),
                  borderRadius: BorderRadius.circular(8),
                  border:       Border.all(color: kCyan),
                ),
                child: const Icon(Icons.send_rounded, color: kCyan, size: 18),
              ),
            ),
          ]),
        ],
      ),
    );
  }

  // ─────────────────────────────────────────
  //  HUD
  // ─────────────────────────────────────────
  Widget _buildHud() {
    return Container(
      padding: EdgeInsets.fromLTRB(
          14, MediaQuery.of(context).padding.top + 6, 10, 10),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin:  Alignment.topCenter,
          end:    Alignment.bottomCenter,
          colors: [Colors.black87, Colors.transparent],
        ),
      ),
      child: Row(children: [
        AnimatedBuilder(
          animation: _pulseAnim,
          builder: (_, __) => Container(
            width: 9, height: 9,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: _connected
                  ? Color.lerp(kCyan, kGreen, _pulseAnim.value)
                  : Colors.white30,
              boxShadow: _connected
                  ? [BoxShadow(color: kCyan, blurRadius: 10, spreadRadius: 3)]
                  : null,
            ),
          ),
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            _statusMsg,
            style: TextStyle(
                color:    _connected ? kCyan : Colors.white54,
                fontSize: 11,
                letterSpacing: 0.5),
            overflow: TextOverflow.ellipsis,
          ),
        ),
        _HudBtn(
          icon:   _ttsEnabled
              ? Icons.volume_up_rounded
              : Icons.volume_off_rounded,
          active: _ttsEnabled,
          onTap: () {
            setState(() => _ttsEnabled = !_ttsEnabled);
            if (!_ttsEnabled) _tts.stop();
          },
        ),
        const SizedBox(width: 6),
        _HudBtn(
          icon:   Icons.tune_rounded,
          active: _showPanel,
          onTap:  _togglePanel,
          badge:  _alerts.isNotEmpty ? '${_alerts.length}' : null,
        ),
        const SizedBox(width: 6),
        GestureDetector(
          onTap: _connected ? _disconnect : _connect,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                  color: _connected
                      ? Colors.redAccent.withOpacity(0.7)
                      : kCyan.withOpacity(0.7)),
              color: _connected
                  ? Colors.redAccent.withOpacity(0.12)
                  : kCyan.withOpacity(0.12),
            ),
            child: Text(
              _connecting ? '…' : (_connected ? 'Disconnect' : 'Connect'),
              style: TextStyle(
                  color:      _connected ? Colors.redAccent : kCyan,
                  fontSize:   11,
                  fontWeight: FontWeight.w700),
            ),
          ),
        ),
      ]),
    );
  }

  // ─────────────────────────────────────────
  //  Alert banner
  // ─────────────────────────────────────────
  Widget _buildBanner(AlertEntry a) {
    final c = a.isCritical ? Colors.redAccent : kCyan;
    return ScaleTransition(
      scale: _bannerAnim,
      child: Container(
        padding:    const EdgeInsets.fromLTRB(16, 12, 12, 12),
        decoration: BoxDecoration(
          color:        Colors.black.withOpacity(0.9),
          borderRadius: BorderRadius.circular(14),
          border:       Border.all(color: c, width: 1.5),
          boxShadow:    [BoxShadow(color: c.withOpacity(0.4), blurRadius: 22)],
        ),
        child: Row(children: [
          Icon(a.isCritical
              ? Icons.warning_amber_rounded
              : Icons.campaign_rounded,
              color: c, size: 22),
          const SizedBox(width: 12),
          Expanded(
            child: Text(a.text,
                style: const TextStyle(
                    color: Colors.white, fontSize: 14,
                    fontWeight: FontWeight.w600, height: 1.3)),
          ),
          GestureDetector(
            onTap: () => setState(() => _banner = null),
            child: const Icon(Icons.close_rounded,
                color: Colors.white38, size: 18),
          ),
        ]),
      ),
    );
  }

  // ─────────────────────────────────────────
  //  Side panel
  // ─────────────────────────────────────────
  Widget _buildSidePanel() {
    return Container(
      decoration: const BoxDecoration(
        color:  Color(0xF00A0E1A),
        border: Border(left: BorderSide(color: Color(0xFF1A2340))),
      ),
      child: Column(children: [
        Container(
          padding: EdgeInsets.fromLTRB(
              10, MediaQuery.of(context).padding.top + 10, 8, 8),
          child: Row(children: [
            _PanelTab(
              label: 'Alerts', count: _alerts.length,
              selected: _panelTab == 0,
              onTap: () => setState(() => _panelTab = 0),
            ),
            const SizedBox(width: 6),
            _PanelTab(
              label: 'Priorities', count: _priorities.length,
              selected: _panelTab == 1,
              onTap: () => setState(() => _panelTab = 1),
            ),
            const SizedBox(width: 6),
            _PanelTab(
              label: 'Voice', count: 0,
              selected: _panelTab == 2,
              onTap: () => setState(() => _panelTab = 2),
            ),
            const SizedBox(width: 6),
            _PanelTab(
              label: 'Settings', count: 0,
              selected: _panelTab == 3,
              onTap: () => setState(() => _panelTab = 3),
            ),
            const Spacer(),
            IconButton(
              icon:        const Icon(Icons.close_rounded,
                  color: Colors.white38, size: 19),
              onPressed:   _togglePanel,
              padding:     EdgeInsets.zero,
              constraints: const BoxConstraints(),
            ),
          ]),
        ),
        const Divider(color: Color(0xFF1A2340), height: 1),
        Expanded(
          child: IndexedStack(
            index: _panelTab,
            children: [
              _buildAlertList(),
              _buildPriorityList(),
              _buildVoicePanel(),
              _buildSettings(),
            ],
          ),
        ),
      ]),
    );
  }

  // ─────────────────────────────────────────
  //  Panel: Alerts
  // ─────────────────────────────────────────
  Widget _buildAlertList() {
    if (_alerts.isEmpty) {
      return const Center(child: Text('No alerts yet',
          style: TextStyle(color: Colors.white24, fontSize: 12)));
    }
    return Column(children: [
      Padding(
        padding: const EdgeInsets.fromLTRB(14, 8, 14, 4),
        child: Row(children: [
          Text('${_alerts.length} alerts',
              style: const TextStyle(color: Colors.white38, fontSize: 10)),
          const Spacer(),
          GestureDetector(
            onTap: () => setState(() => _alerts.clear()),
            child: const Text('CLEAR',
                style: TextStyle(color: Colors.white30, fontSize: 9,
                    letterSpacing: 0.8)),
          ),
        ]),
      ),
      Expanded(
        child: ListView.builder(
          padding:     const EdgeInsets.symmetric(horizontal: 10),
          itemCount:   _alerts.length,
          itemBuilder: (_, i) => _AlertTile(alert: _alerts[i]),
        ),
      ),
    ]);
  }

  // ─────────────────────────────────────────
  //  Panel: Priorities
  // ─────────────────────────────────────────
  Widget _buildPriorityList() {
    return Column(children: [
      Padding(
        padding: const EdgeInsets.fromLTRB(14, 12, 14, 8),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('Custom priorities',
              style: TextStyle(
                  color: kCyan, fontSize: 11, fontWeight: FontWeight.w700)),
          const SizedBox(height: 4),
          const Text(
            'Use the mic button or Voice tab to add items.',
            style: TextStyle(color: Colors.white30, fontSize: 10, height: 1.5),
          ),
        ]),
      ),
      Expanded(
        child: _priorities.isEmpty
            ? const Center(child: Text('No custom priorities',
                style: TextStyle(color: Colors.white24, fontSize: 12)))
            : ListView.builder(
                padding:     const EdgeInsets.symmetric(horizontal: 10),
                itemCount:   _priorities.length,
                itemBuilder: (_, i) => Container(
                  margin:  const EdgeInsets.only(bottom: 6),
                  padding: const EdgeInsets.symmetric(
                      horizontal: 12, vertical: 9),
                  decoration: BoxDecoration(
                    color:        const Color(0xFF0D1428),
                    borderRadius: BorderRadius.circular(8),
                    border:       Border.all(
                        color: kCyan.withOpacity(0.18)),
                  ),
                  child: Row(children: [
                    const Icon(Icons.bookmark_rounded,
                        color: kCyan, size: 13),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(_priorities[i],
                          style: const TextStyle(
                              color: Colors.white, fontSize: 12)),
                    ),
                    // Quick remove via swipe-style tap
                    GestureDetector(
                      onTap: () =>
                          _sendVoiceCommand('remove ${_priorities[i]}'),
                      child: const Icon(Icons.remove_circle_outline_rounded,
                          color: Colors.white24, size: 14),
                    ),
                  ]),
                ),
              ),
      ),
    ]);
  }

  // ─────────────────────────────────────────
  //  Panel: Voice (NEW)
  // ─────────────────────────────────────────
  Widget _buildVoicePanel() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Status header
          Row(children: [
            Container(
              width: 8, height: 8,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _sttAvailable ? kGreen : Colors.redAccent,
              ),
            ),
            const SizedBox(width: 8),
            Text(
              _sttAvailable
                  ? 'Microphone ready'
                  : 'Microphone unavailable',
              style: TextStyle(
                  color: _sttAvailable ? kGreen : Colors.redAccent,
                  fontSize: 11,
                  fontWeight: FontWeight.w600),
            ),
          ]),
          const SizedBox(height: 16),

          // Big mic button centered
          Center(
            child: GestureDetector(
              onLongPressStart: (_) => _startListening(),
              onLongPressEnd:   (_) => _stopListening(send: true),
              onTap:            _isListening
                  ? () => _stopListening(send: true)
                  : _startListening,
              child: AnimatedBuilder(
                animation: _micPulseAnim,
                builder: (_, child) => Transform.scale(
                  scale: _isListening ? _micPulseAnim.value : 1.0,
                  child: child,
                ),
                child: Container(
                  width: 80, height: 80,
                  decoration: BoxDecoration(
                    shape:  BoxShape.circle,
                    color:  _isListening
                        ? Colors.redAccent.withOpacity(0.9)
                        : kCyan.withOpacity(0.12),
                    border: Border.all(
                        color: _isListening ? Colors.redAccent : kCyan,
                        width: 2),
                    boxShadow: [
                      BoxShadow(
                        color: (_isListening
                            ? Colors.redAccent
                            : kCyan).withOpacity(0.4),
                        blurRadius:   20,
                        spreadRadius: 2,
                      ),
                    ],
                  ),
                  child: Icon(
                    _isListening ? Icons.stop_rounded : Icons.mic_rounded,
                    color: _isListening ? Colors.white : kCyan,
                    size: 36,
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(height: 8),
          Center(
            child: Text(
              _isListening
                  ? '🔴  Listening…  tap to stop'
                  : 'Tap or hold to speak',
              style: const TextStyle(color: Colors.white54, fontSize: 11),
            ),
          ),

          // Live transcript
          if (_isListening && _liveText.isNotEmpty) ...[
            const SizedBox(height: 12),
            Container(
              width:      double.infinity,
              padding:    const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color:        Colors.white.withOpacity(0.04),
                borderRadius: BorderRadius.circular(8),
                border:       Border.all(color: kCyan.withOpacity(0.3)),
              ),
              child: Text(
                _liveText,
                style: const TextStyle(color: Colors.white70, fontSize: 13),
              ),
            ),
          ],

          const SizedBox(height: 24),
          const Divider(color: Colors.white12),
          const SizedBox(height: 16),

          // Manual text command entry
          const Text('TYPE A COMMAND',
              style: TextStyle(
                  color:        kCyan,
                  fontSize:     9,
                  fontWeight:   FontWeight.w700,
                  letterSpacing: 1.2)),
          const SizedBox(height: 6),
          Text(
            '"add pharmacy"  •  "remove hospital"  •  "list"  •  "clear"',
            style: TextStyle(
                color: Colors.white30, fontSize: 9, height: 1.5),
          ),
          const SizedBox(height: 10),
          Row(children: [
            Expanded(
              child: TextField(
                controller: _cmdCtrl,
                style:      const TextStyle(color: Colors.white, fontSize: 13),
                decoration: InputDecoration(
                  hintText:       'Type command...',
                  hintStyle:      TextStyle(color: Colors.white30, fontSize: 12),
                  contentPadding: const EdgeInsets.symmetric(
                      horizontal: 10, vertical: 8),
                  filled:     true,
                  fillColor:  Colors.white.withOpacity(0.05),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                    borderSide:  BorderSide(color: kCyan.withOpacity(0.3)),
                  ),
                  enabledBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                    borderSide:  BorderSide(color: kCyan.withOpacity(0.2)),
                  ),
                  focusedBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                    borderSide:  const BorderSide(color: kCyan),
                  ),
                ),
                onSubmitted: _submitTextCommand,
              ),
            ),
            const SizedBox(width: 8),
            GestureDetector(
              onTap: () => _submitTextCommand(_cmdCtrl.text),
              child: Container(
                width: 42, height: 42,
                decoration: BoxDecoration(
                  color:        kCyan.withOpacity(0.15),
                  borderRadius: BorderRadius.circular(8),
                  border:       Border.all(color: kCyan),
                ),
                child: const Icon(Icons.send_rounded, color: kCyan, size: 18),
              ),
            ),
          ]),

          const SizedBox(height: 24),
          const Divider(color: Colors.white12),
          const SizedBox(height: 12),

          // Last command sent
          if (_lastCommand.isNotEmpty) ...[
            const Text('LAST SENT',
                style: TextStyle(
                    color: Colors.white30, fontSize: 9,
                    fontWeight: FontWeight.w600, letterSpacing: 1.0)),
            const SizedBox(height: 6),
            Container(
              width:      double.infinity,
              padding:    const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
              decoration: BoxDecoration(
                color:        Colors.white.withOpacity(0.03),
                borderRadius: BorderRadius.circular(8),
                border:       Border.all(color: Colors.white12),
              ),
              child: Text(
                '"$_lastCommand"',
                style: const TextStyle(
                    color: Colors.white70, fontSize: 12,
                    fontStyle: FontStyle.italic),
              ),
            ),
          ],

          const SizedBox(height: 20),

          // Quick-action chips
          const Text('QUICK COMMANDS',
              style: TextStyle(
                  color: Colors.white30, fontSize: 9,
                  fontWeight: FontWeight.w600, letterSpacing: 1.0)),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8, runSpacing: 8,
            children: [
              _QuickCmd(label: 'List priorities',
                  onTap: () => _sendVoiceCommand('list')),
              _QuickCmd(label: 'Clear all',
                  onTap: () => _sendVoiceCommand('clear my priorities')),
              ..._priorities.map((p) => _QuickCmd(
                    label: 'Remove $p',
                    onTap: () => _sendVoiceCommand('remove $p'),
                    color: Colors.redAccent,
                  )),
            ],
          ),
        ],
      ),
    );
  }

  void _submitTextCommand(String val) {
    final t = val.trim();
    if (t.isNotEmpty) {
      _sendVoiceCommand(t);
      _cmdCtrl.clear();
    }
  }

  // ─────────────────────────────────────────
  //  Panel: Settings
  // ─────────────────────────────────────────
  Widget _buildSettings() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            const Icon(Icons.language_rounded, color: kCyan, size: 15),
            const SizedBox(width: 6),
            const Text('Language',
                style: TextStyle(
                    color: kCyan, fontSize: 12,
                    fontWeight: FontWeight.w700, letterSpacing: 0.6)),
          ]),
          const SizedBox(height: 4),
          Text(
            _connected
                ? 'Changes are sent to the server immediately.'
                : 'Connect first to sync language with the server.',
            style: TextStyle(color: Colors.white30, fontSize: 9, height: 1.5),
          ),
          const SizedBox(height: 16),
          const Text('Voice input (STT)',
              style: TextStyle(color: Colors.white54, fontSize: 10,
                  fontWeight: FontWeight.w600, letterSpacing: 0.4)),
          const SizedBox(height: 6),
          _LangDropdown(
            value:    _sourceLanguage,
            items:    kSourceLanguages,
            hint:     'Select voice language',
            onChanged: (val) { if (val != null) _sendLanguage(source: val); },
          ),
          const SizedBox(height: 6),
          Text(
            'Language used to recognise voice commands (on-device mic and server mic).',
            style: TextStyle(color: Colors.white24, fontSize: 9, height: 1.4),
          ),
          const SizedBox(height: 20),
          const Text('Speech output (TTS)',
              style: TextStyle(color: Colors.white54, fontSize: 10,
                  fontWeight: FontWeight.w600, letterSpacing: 0.4)),
          const SizedBox(height: 6),
          _LangDropdown(
            value:    _targetLanguage,
            items:    kTargetLanguages,
            hint:     'Select output language',
            onChanged: (val) { if (val != null) _sendLanguage(target: val); },
          ),
          const SizedBox(height: 6),
          Text(
            'Alerts are translated and spoken in this language on both server and device.',
            style: TextStyle(color: Colors.white24, fontSize: 9, height: 1.4),
          ),
          const SizedBox(height: 24),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color:        Colors.white.withOpacity(0.03),
              borderRadius: BorderRadius.circular(8),
              border:       Border.all(color: Colors.white12),
            ),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              _StateRow(label: 'STT',
                  value: kSourceLanguages[_sourceLanguage] ?? _sourceLanguage),
              const SizedBox(height: 6),
              _StateRow(label: 'TTS',
                  value: kTargetLanguages[_targetLanguage] ?? _targetLanguage),
              const SizedBox(height: 6),
              _StateRow(label: 'Device TTS locale',
                  value: kTtsLocale[_targetLanguage] ?? 'en-US'),
              const SizedBox(height: 6),
              _StateRow(label: 'STT available',
                  value: _sttAvailable ? 'Yes' : 'No'),
            ]),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────
//  CONSTANTS
// ─────────────────────────────────────────────────────────────
const kCyan  = Color(0xFF00E5FF);
const kGreen = Color(0xFF00FF88);

// ─────────────────────────────────────────────────────────────
//  HUD BUTTON
// ─────────────────────────────────────────────────────────────
class _HudBtn extends StatelessWidget {
  final IconData icon;
  final bool active;
  final VoidCallback onTap;
  final String? badge;

  const _HudBtn({
    required this.icon,
    required this.active,
    required this.onTap,
    this.badge,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Stack(clipBehavior: Clip.none, children: [
        Container(
          width: 34, height: 34,
          decoration: BoxDecoration(
            shape:  BoxShape.circle,
            color:  active ? kCyan.withOpacity(0.16) : Colors.white10,
            border: Border.all(
                color: active ? kCyan.withOpacity(0.55) : Colors.white12),
          ),
          child: Icon(icon,
              color: active ? kCyan : Colors.white38, size: 17),
        ),
        if (badge != null)
          Positioned(
            top: -4, right: -4,
            child: Container(
              padding: const EdgeInsets.all(3),
              decoration: const BoxDecoration(
                  color: Colors.redAccent, shape: BoxShape.circle),
              child: Text(badge!,
                  style: const TextStyle(
                      color:      Colors.white,
                      fontSize:   8,
                      fontWeight: FontWeight.w700)),
            ),
          ),
      ]),
    );
  }
}

// ─────────────────────────────────────────────────────────────
//  CHIP
// ─────────────────────────────────────────────────────────────
class _Chip extends StatelessWidget {
  final IconData icon;
  final String   label;
  final Color    color;

  const _Chip({
    required this.icon,
    required this.label,
    required this.color,
  });

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color:        Colors.black54,
          borderRadius: BorderRadius.circular(20),
          border:       Border.all(color: color.withOpacity(0.4)),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(icon, color: color, size: 12),
          const SizedBox(width: 5),
          Text(label,
              style: TextStyle(
                  color: color, fontSize: 10, fontWeight: FontWeight.w600)),
        ]),
      );
}

// ─────────────────────────────────────────────────────────────
//  CONNECT BUTTON (kept for reference, unused in current layout)
// ─────────────────────────────────────────────────────────────
class _ConnectBtn extends StatelessWidget {
  final String       label;
  final VoidCallback onTap;
  const _ConnectBtn({required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) => GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 26, vertical: 13),
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(30),
            border:       Border.all(color: kCyan, width: 1.5),
            color:        kCyan.withOpacity(0.08),
            boxShadow: [BoxShadow(
                color:      kCyan.withOpacity(0.28),
                blurRadius: 24, spreadRadius: 2)],
          ),
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            const Icon(Icons.play_arrow_rounded, color: kCyan, size: 20),
            const SizedBox(width: 8),
            Text(label,
                style: const TextStyle(
                    color:         kCyan,
                    fontSize:      14,
                    fontWeight:    FontWeight.w700,
                    letterSpacing: 0.4)),
          ]),
        ),
      );
}

// ─────────────────────────────────────────────────────────────
//  PANEL TAB
// ─────────────────────────────────────────────────────────────
class _PanelTab extends StatelessWidget {
  final String     label;
  final int        count;
  final bool       selected;
  final VoidCallback onTap;

  const _PanelTab({
    required this.label,
    required this.count,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) => GestureDetector(
        onTap: onTap,
        child: AnimatedContainer(
          duration:  const Duration(milliseconds: 180),
          padding:   const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          decoration: BoxDecoration(
            color:        selected
                ? kCyan.withOpacity(0.13)
                : Colors.transparent,
            borderRadius: BorderRadius.circular(8),
            border:       Border.all(
                color: selected
                    ? kCyan.withOpacity(0.45)
                    : Colors.transparent),
          ),
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            Text(label,
                style: TextStyle(
                    color:      selected ? kCyan : Colors.white38,
                    fontSize:   10,
                    fontWeight: FontWeight.w600)),
            if (count > 0) ...[
              const SizedBox(width: 4),
              Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 4, vertical: 1),
                decoration: BoxDecoration(
                    color: selected
                        ? kCyan.withOpacity(0.2)
                        : Colors.white10,
                    borderRadius: BorderRadius.circular(8)),
                child: Text('$count',
                    style: TextStyle(
                        color:      selected ? kCyan : Colors.white38,
                        fontSize:   8,
                        fontWeight: FontWeight.w700)),
              ),
            ],
          ]),
        ),
      );
}

// ─────────────────────────────────────────────────────────────
//  ALERT TILE
// ─────────────────────────────────────────────────────────────
class _AlertTile extends StatelessWidget {
  final AlertEntry alert;
  const _AlertTile({required this.alert});

  String _fmt(DateTime dt) =>
      '${dt.hour.toString().padLeft(2, '0')}:'
      '${dt.minute.toString().padLeft(2, '0')}:'
      '${dt.second.toString().padLeft(2, '0')}';

  @override
  Widget build(BuildContext context) {
    final c = alert.isCritical ? Colors.redAccent : kCyan;
    return Container(
      margin:  const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color:        const Color(0xFF0D1428),
        borderRadius: BorderRadius.circular(8),
        border:       Border.all(
            color: c.withOpacity(alert.isCritical ? 0.4 : 0.15)),
      ),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Icon(
          alert.isCritical
              ? Icons.warning_amber_rounded
              : Icons.campaign_rounded,
          color: c, size: 15,
        ),
        const SizedBox(width: 8),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(alert.text,
                  style: const TextStyle(
                      color: Colors.white, fontSize: 11, height: 1.3)),
              const SizedBox(height: 3),
              Text(_fmt(alert.ts),
                  style: const TextStyle(
                      color: Colors.white30, fontSize: 9)),
            ],
          ),
        ),
      ]),
    );
  }
}

// ─────────────────────────────────────────────────────────────
//  QUICK COMMAND CHIP
// ─────────────────────────────────────────────────────────────
class _QuickCmd extends StatelessWidget {
  final String       label;
  final VoidCallback onTap;
  final Color        color;

  const _QuickCmd({
    required this.label,
    required this.onTap,
    this.color = kCyan,
  });

  @override
  Widget build(BuildContext context) => GestureDetector(
        onTap: onTap,
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
          decoration: BoxDecoration(
            color:        color.withOpacity(0.08),
            borderRadius: BorderRadius.circular(20),
            border:       Border.all(color: color.withOpacity(0.4)),
          ),
          child: Text(label,
              style: TextStyle(
                  color:      color,
                  fontSize:   10,
                  fontWeight: FontWeight.w600)),
        ),
      );
}

// ─────────────────────────────────────────────────────────────
//  SETTINGS HELPERS
// ─────────────────────────────────────────────────────────────
class _LangDropdown extends StatelessWidget {
  final String                value;
  final Map<String, String>   items;
  final String                hint;
  final ValueChanged<String?> onChanged;

  const _LangDropdown({
    required this.value,
    required this.items,
    required this.hint,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 2),
        decoration: BoxDecoration(
          color:        Colors.white.withOpacity(0.04),
          borderRadius: BorderRadius.circular(8),
          border:       Border.all(color: kCyan.withOpacity(0.3)),
        ),
        child: DropdownButton<String>(
          value:            items.containsKey(value) ? value : null,
          hint:             Text(hint,
              style: const TextStyle(color: Colors.white30, fontSize: 12)),
          isExpanded:       true,
          underline:        const SizedBox(),
          dropdownColor:    const Color(0xFF0D1428),
          iconEnabledColor: kCyan,
          style:            const TextStyle(color: Colors.white, fontSize: 12),
          items: items.entries
              .map((e) => DropdownMenuItem(value: e.key, child: Text(e.value)))
              .toList(),
          onChanged: onChanged,
        ),
      );
}

class _StateRow extends StatelessWidget {
  final String label;
  final String value;
  const _StateRow({required this.label, required this.value});

  @override
  Widget build(BuildContext context) => Row(children: [
        Text('$label  ',
            style: const TextStyle(
                color: Colors.white30, fontSize: 9,
                fontWeight: FontWeight.w600)),
        Expanded(
          child: Text(value,
              style: const TextStyle(
                  color: kCyan, fontSize: 9, fontWeight: FontWeight.w700),
              overflow: TextOverflow.ellipsis),
        ),
      ]);
}