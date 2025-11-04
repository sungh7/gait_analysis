# iOS Deployment Guide - V7 Pure 3D Gait Analysis

## Overview

Complete guide for deploying the V7 Pure 3D gait analysis algorithm on iOS devices using Flutter and MediaPipe.

## Prerequisites

### Development Environment
- macOS 12.0+ (Monterey or later)
- Xcode 14.0+ with Command Line Tools
- CocoaPods 1.11+
- Flutter 3.10+
- iOS deployment target: 14.0+

### Install CocoaPods
```bash
sudo gem install cocoapods
pod setup
```

### Install Flutter
```bash
# Follow official Flutter installation guide
# https://docs.flutter.dev/get-started/install/macos
flutter doctor
```

## Project Setup

### 1. Navigate to iOS Directory
```bash
cd gait_analysis_mobile_app/ios
```

### 2. Install Dependencies
```bash
# Install CocoaPods dependencies
pod install

# If you encounter issues, try:
pod repo update
pod install --repo-update
```

### 3. Open Xcode Project
```bash
# Always use .xcworkspace, NOT .xcodeproj
open Runner.xcworkspace
```

## MediaPipe Integration

### Architecture
```
Flutter App (Dart)
    ↓
MethodChannel: 'gait_analysis/mediapipe'
    ↓
MediaPipePlugin.swift (iOS Native)
    ↓
MediaPipe Pose Landmarker (C++)
    ↓
33 3D World Landmarks
    ↓
V7Pure3DService.dart
    ↓
Detection Result
```

### Key Files

1. **MediaPipePlugin.swift** (`ios/Runner/MediaPipePlugin.swift`)
   - Native iOS MediaPipe implementation
   - Handles pose detection in VIDEO mode
   - Returns 33 3D world landmarks
   - Performance: ~30fps on iPhone 11+

2. **AppDelegate.swift** (`ios/Runner/AppDelegate.swift`)
   - Registers MediaPipe plugin
   - Application lifecycle management

3. **Podfile** (`ios/Podfile`)
   - MediaPipeTasksVision dependency
   - iOS 14.0 minimum deployment target

4. **Info.plist** (`ios/Runner/Info.plist`)
   - Camera permission: `NSCameraUsageDescription`
   - Motion permission: `NSMotionUsageDescription`

## MediaPipe Model Setup

### Download Model
```bash
# Create assets directory
mkdir -p ios/Runner/Assets

# Download pose_landmarker.task model
cd ios/Runner/Assets
curl -O https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

# Rename to expected filename
mv pose_landmarker_heavy.task pose_landmarker.task
```

### Add to Xcode
1. Open `Runner.xcworkspace` in Xcode
2. Right-click on `Runner` folder
3. Select "Add Files to Runner..."
4. Navigate to `Assets/pose_landmarker.task`
5. Check "Copy items if needed"
6. Check "Runner" target
7. Click "Add"

## Build Configuration

### Xcode Settings

#### General Tab
- **Display Name**: Gait Analysis
- **Bundle Identifier**: com.gaitanalysis.app
- **Version**: 1.0.0
- **Build**: 1
- **Deployment Target**: iOS 14.0
- **Devices**: iPhone, iPad

#### Signing & Capabilities
- **Automatically manage signing**: ✓
- **Team**: (Select your Apple Developer Team)
- **Signing Certificate**: Apple Development

#### Build Settings
- **Swift Language Version**: Swift 5
- **Enable Bitcode**: No (MediaPipe requires)
- **Optimization Level (Release)**: Optimize for Size [-Osize]

### Camera Permissions

The app requires camera access. Permissions are configured in `Info.plist`:

```xml
<key>NSCameraUsageDescription</key>
<string>This app needs camera access to analyze your gait patterns using real-time video.</string>

<key>NSMotionUsageDescription</key>
<string>This app uses motion sensors to improve gait analysis accuracy.</string>
```

## Building the App

### Debug Build (Simulator)
```bash
# From project root
flutter run -d "iPhone 14 Pro"

# Or build explicitly
flutter build ios --debug --simulator
```

### Debug Build (Physical Device)
```bash
# Connect iPhone via USB
flutter devices

# Run on connected device
flutter run -d <device-id>
```

### Release Build
```bash
# Build release IPA
flutter build ipa --release

# Output location:
# build/ios/archive/Runner.xcarchive
```

## Testing

### Unit Tests
```bash
# Run Dart unit tests
flutter test

# Run iOS native tests
cd ios
xcodebuild test \
  -workspace Runner.xcworkspace \
  -scheme Runner \
  -destination 'platform=iOS Simulator,name=iPhone 14 Pro'
```

### Performance Testing

Expected performance on real devices:

| Device | FPS | Inference Time | Memory |
|--------|-----|----------------|--------|
| iPhone 14 Pro | 30 fps | 25-30ms | ~200MB |
| iPhone 13 | 30 fps | 30-35ms | ~180MB |
| iPhone 12 | 28 fps | 35-40ms | ~180MB |
| iPhone 11 | 25 fps | 40-50ms | ~160MB |

### Testing Checklist
- [ ] Camera preview works correctly
- [ ] Pose landmarks display in real-time
- [ ] 6-second recording completes successfully
- [ ] V7 analysis produces results (68.2% expected accuracy)
- [ ] No memory leaks (use Xcode Instruments)
- [ ] Stable frame rate (25-30 fps)
- [ ] App works in portrait and landscape modes

## App Store Submission

### Prepare for Submission

1. **Update Version**
   ```yaml
   # pubspec.yaml
   version: 1.0.0+1
   ```

2. **Create App Icons**
   - Generate all required sizes (1024x1024 down to 20x20)
   - Place in `ios/Runner/Assets.xcassets/AppIcon.appiconset/`

3. **Screenshots**
   - 6.7" display (iPhone 14 Pro Max): 1290 x 2796
   - 6.5" display (iPhone 11 Pro Max): 1242 x 2688
   - 5.5" display (iPhone 8 Plus): 1242 x 2208
   - 12.9" iPad Pro: 2048 x 2732

4. **App Store Connect**
   - Create new app in App Store Connect
   - Fill in metadata (description, keywords, category)
   - Upload screenshots
   - Set pricing

### Build and Upload

```bash
# Clean build
flutter clean
flutter pub get

# Build release archive
flutter build ipa --release

# Upload to App Store Connect
# Option 1: Use Xcode Organizer
open build/ios/archive/Runner.xcarchive

# Option 2: Use Transporter app
# Download from Mac App Store
# Drag .ipa file to Transporter
```

### App Review Guidelines

Ensure compliance with Apple's guidelines:
- Privacy policy (camera usage explanation)
- Medical disclaimer (if applicable)
- Accurate app description
- No misleading health claims
- HIPAA compliance (if storing health data)

## Troubleshooting

### Common Issues

#### 1. Pod Install Fails
```bash
# Update CocoaPods
sudo gem install cocoapods

# Clear pod cache
rm -rf ~/Library/Caches/CocoaPods
pod cache clean --all

# Re-install
pod deintegrate
pod install
```

#### 2. MediaPipe Model Not Found
```
Error: Model file not found
```
**Solution**: Ensure `pose_landmarker.task` is added to Xcode project as a resource.

#### 3. Camera Permission Denied
```
Error: Camera access denied
```
**Solution**: Check Info.plist has `NSCameraUsageDescription` and request permissions at runtime.

#### 4. Slow Performance
```
FPS < 20 on iPhone 11+
```
**Solutions**:
- Enable GPU acceleration in MediaPipePlugin
- Reduce camera resolution to 640x480
- Sample every other frame (30fps → 15fps)

#### 5. Build Failed - Swift Version
```
Error: Swift version mismatch
```
**Solution**: Set Swift Language Version to 5.0 in Build Settings.

## Performance Optimization

### GPU Acceleration
```swift
// MediaPipePlugin.swift
options.baseOptions.delegate = .GPU  // Enable Metal acceleration
```

### Frame Sampling
```dart
// realtime_camera_service.dart
final shouldProcess = _frameCount % 2 == 0;  // Process every 2nd frame
```

### Camera Resolution
```dart
// Initialize camera with lower resolution
ResolutionPreset.medium  // 640x480 instead of high
```

### Memory Management
```swift
// Dispose detector when not in use
override func applicationDidEnterBackground(_ application: UIApplication) {
    mediaPipePlugin?.dispose()
}
```

## Known Limitations

1. **iOS Simulator**: MediaPipe may have reduced performance on simulator. Test on real devices.
2. **Metal GPU**: Requires iOS 14.0+ for optimal GPU acceleration.
3. **Background Processing**: Video detection pauses when app enters background.
4. **Model Size**: pose_landmarker.task is ~27MB, increases app size.

## V7 Pure 3D Algorithm Performance

Expected results on iOS (same as Android):

- **Overall Accuracy**: 68.2%
- **Overall Sensitivity**: 92.2%
- **Clinical Pathology Sensitivity**: 98.6%
- **Parkinsons Detection**: 100% (6/6)
- **Stroke Detection**: 100% (11/11)
- **Cerebral Palsy Detection**: 100% (24/24)

## Support

For iOS-specific issues:
1. Check Xcode console for detailed error messages
2. Use Instruments to profile performance
3. Review Apple Developer forums
4. Check MediaPipe iOS documentation

## Next Steps

- [ ] Complete iOS testing on 5+ devices
- [ ] Optimize performance to 30fps on all devices
- [ ] Implement background processing (if needed)
- [ ] Add TestFlight beta testing
- [ ] Submit to App Store
- [ ] Collect user feedback
- [ ] Iterate and improve

## References

- [Flutter iOS Deployment](https://docs.flutter.dev/deployment/ios)
- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [Apple Developer Guidelines](https://developer.apple.com/app-store/review/guidelines/)
- [CocoaPods](https://cocoapods.org/)
