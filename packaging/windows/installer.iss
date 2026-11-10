; Compile with /DAppVersion=<package version> /DBundleDir=<absolute folder>
; /DOutputDir=<absolute folder>. build.ps1 supplies all three.
#ifndef AppVersion
  #error AppVersion is required
#endif
#ifndef BundleDir
  #error BundleDir is required
#endif
#ifndef OutputDir
  #error OutputDir is required
#endif

[Setup]
AppId={{6E3A06D8-C3AC-4D6D-A72B-AC37EC33867D}
AppName=napari-moltrack
AppVersion={#AppVersion}
AppPublisher=MolTrack
AppPublisherURL=https://github.com/piedrro/napari-moltrack
DefaultDirName={localappdata}\Programs\napari-moltrack
DefaultGroupName=napari-moltrack
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
MinVersion=10.0.17763
OutputDir={#OutputDir}
OutputBaseFilename=napari-moltrack-Setup-{#AppVersion}-x64
SetupIconFile=moltrack.ico
UninstallDisplayIcon={app}\napari-moltrack.exe
LicenseFile=..\..\LICENSE
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
SetupLogging=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; Remove stale libraries on upgrade. User models, logs and projects live elsewhere.
Type: filesandordirs; Name: "{app}\_internal"
Type: files; Name: "{app}\MolTrack.exe"
Type: files; Name: "{userprograms}\MolTrack\MolTrack.lnk"
Type: files; Name: "{userprograms}\MolTrack\MolTrack logs.lnk"
Type: files; Name: "{autodesktop}\MolTrack.lnk"

[Icons]
Name: "{group}\napari-moltrack"; Filename: "{app}\napari-moltrack.exe"; WorkingDir: "{app}"
Name: "{group}\napari-moltrack logs"; Filename: "{win}\explorer.exe"; Parameters: """{localappdata}\MolTrack\logs"""
Name: "{autodesktop}\napari-moltrack"; Filename: "{app}\napari-moltrack.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\napari-moltrack.exe"; Description: "Launch napari-moltrack"; Flags: nowait postinstall skipifsilent
