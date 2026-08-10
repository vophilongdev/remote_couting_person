import ctypes
import os
import time
import threading
import logging

logger = logging.getLogger("HikWrapper")

_sdk_lock = threading.RLock()
_sdk_instance_count = 0
_sdk_initialized = False

_ERROR_NAMES = {
    1: "username or password error",
    2: "insufficient user authority",
    3: "SDK not initialized",
    4: "channel number error",
    5: "device connection limit reached",
    6: "SDK/device version mismatch",
    7: "network connection failed",
    8: "network send failed",
    9: "network receive failed",
    10: "network receive timeout",
    11: "invalid network response data",
    12: "invalid API call order",
    13: "operation not permitted",
    14: "command timeout",
    17: "invalid parameter/ABI",
    24: "device busy",
    28: "device resource exhausted",
    29: "device operation failed",
    36: "previous operation is still running",
}


def _error_text(code):
    return _ERROR_NAMES.get(int(code), "unknown HCNetSDK error")

# Callback types
# void CALLBACK RealDataCallBack_V30(LONG lPlayHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, void* pUser)
# SDK LONG = 32-bit on all platforms; use c_int, not c_long (which is 64-bit on Linux).
fRealDataCallBack_V30 = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int,       # lPlayHandle  (SDK LONG = 32-bit)
    ctypes.c_uint,      # dwDataType
    ctypes.POINTER(ctypes.c_ubyte),  # pBuffer
    ctypes.c_uint,      # dwBufSize
    ctypes.c_void_p     # pUser
)

# Structs used by this wrapper.
class NET_DVR_DEVICEINFO_V30(ctypes.Structure):
    _fields_ = [
        ("sSerialNumber", ctypes.c_char * 48),
        ("byAlarmInPortNum", ctypes.c_ubyte),
        ("byAlarmOutPortNum", ctypes.c_ubyte),
        ("byDiskNum", ctypes.c_ubyte),
        ("byDVRType", ctypes.c_ubyte),
        ("byChanNum", ctypes.c_ubyte),
        ("byStartChan", ctypes.c_ubyte),
        ("byAudioChanNum", ctypes.c_ubyte),
        ("byIPChanNum", ctypes.c_ubyte),
        ("byZeroChanNum", ctypes.c_ubyte),
        ("byMainProto", ctypes.c_ubyte),
        ("bySubProto", ctypes.c_ubyte),
        ("bySupport", ctypes.c_ubyte),
        ("bySupport1", ctypes.c_ubyte),
        ("bySupport2", ctypes.c_ubyte),
        ("wDevType", ctypes.c_ushort),
        ("bySupport3", ctypes.c_ubyte),
        ("byMultiStreamProto", ctypes.c_ubyte),
        ("byStartDChan", ctypes.c_ubyte),
        ("byStartDTalkChan", ctypes.c_ubyte),
        ("byHighDChanNum", ctypes.c_ubyte),
        ("bySupport4", ctypes.c_ubyte),
        ("byLanguageType", ctypes.c_ubyte),
        ("byVoiceInChanNum", ctypes.c_ubyte),
        ("byStartVoiceInChanNo", ctypes.c_ubyte),
        ("bySupport5", ctypes.c_ubyte),
        ("bySupport6", ctypes.c_ubyte),
        ("byMirrorChanNum", ctypes.c_ubyte),
        ("wStartMirrorChanNo", ctypes.c_ushort),
        ("bySupport7", ctypes.c_ubyte),
        ("byRes2", ctypes.c_ubyte),
    ]

class NET_DVR_PREVIEWINFO(ctypes.Structure):
    """Exact ABI from the HCNetSDK 6.1.9.4 header shipped with this project."""
    _fields_ = [
        ("lChannel", ctypes.c_int),
        ("dwStreamType", ctypes.c_uint),
        ("dwLinkMode", ctypes.c_uint),
        # On this Linux SDK HWND is an unsigned 32-bit integer, not a native
        # pointer (confirmed against sizeof/offsetof from the bundled header).
        ("hPlayWnd", ctypes.c_uint),
        ("bBlocked", ctypes.c_uint),
        ("bPassbackRecord", ctypes.c_uint),
        ("byPreviewMode", ctypes.c_ubyte),
        ("byStreamID", ctypes.c_ubyte * 32),
        ("byProtoType", ctypes.c_ubyte),
        ("byRes1", ctypes.c_ubyte),
        ("byVideoCodingType", ctypes.c_ubyte),
        ("dwDisplayBufNum", ctypes.c_uint),
        ("byNPQMode", ctypes.c_ubyte),
        ("byRecvMetaData", ctypes.c_ubyte),
        ("byDataType", ctypes.c_ubyte),
        ("byRes", ctypes.c_ubyte * 213),
    ]


class NET_DVR_LOCAL_SDK_PATH(ctypes.Structure):
    _fields_ = [
        ("sPath", ctypes.c_char * 256),
        ("byRes", ctypes.c_ubyte * 128),
    ]

class HikSDK:
    def __init__(self, lib_path=None):
        self.lib = None
        self.logged_in = False
        self.lUserID = -1
        self.play_handle = -1
        self.callback_ref = None
        self.mock_thread = None
        self.mock_running = False
        self.start_chan = 1
        self.channel_count = 0
        self.is_digital_channel = False
        self._owns_sdk_reference = False
        
        if lib_path is None:
            sdk_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "EN-HCNetSDKV6.1.9.4_build20220412_linux64",
                "EN-HCNetSDKV6.1.9.4_build20220412_linux64",
            )
            lib_path = os.path.join(sdk_root, "lib", "libhcnetsdk.so")
            
        try:
            # Load with RTLD_GLOBAL to resolve sub-dependencies like libhpr, libHCCore
            lib_dir = os.path.dirname(lib_path)
            # Add lib_dir to environment PATH/LD_LIBRARY_PATH if needed or load dependent libraries
            # In Linux, setting ctypes.CDLL load order helps, or we can load HCCore first
            hccore_path = os.path.join(lib_dir, "libHCCore.so")
            hpr_path = os.path.join(lib_dir, "libhpr.so")
            
            if os.path.exists(hpr_path):
                ctypes.CDLL(hpr_path, mode=ctypes.RTLD_GLOBAL)
            if os.path.exists(hccore_path):
                ctypes.CDLL(hccore_path, mode=ctypes.RTLD_GLOBAL)
                
            self.lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            logger.info("Successfully loaded Hikvision HCNetSDK library.")
            
            # Setup signatures
            self.lib.NET_DVR_Init.argtypes = []
            self.lib.NET_DVR_Init.restype = ctypes.c_bool
            
            self.lib.NET_DVR_Cleanup.argtypes = []
            self.lib.NET_DVR_Cleanup.restype = ctypes.c_bool
            
            self.lib.NET_DVR_Login_V30 = getattr(self.lib, "NET_DVR_Login_V30", None)
            if self.lib.NET_DVR_Login_V30:
                self.lib.NET_DVR_Login_V30.argtypes = [
                    ctypes.c_char_p, ctypes.c_ushort, ctypes.c_char_p, ctypes.c_char_p,
                    ctypes.POINTER(NET_DVR_DEVICEINFO_V30)
                ]
                self.lib.NET_DVR_Login_V30.restype = ctypes.c_int
                
            self.lib.NET_DVR_Logout.argtypes = [ctypes.c_int]
            self.lib.NET_DVR_Logout.restype = ctypes.c_bool
            
            self.lib.NET_DVR_RealPlay_V40.argtypes = [
                ctypes.c_int, ctypes.POINTER(NET_DVR_PREVIEWINFO), ctypes.c_void_p, ctypes.c_void_p
            ]
            # SDK LONG is 32-bit even on Linux 64-bit; use c_int to preserve the
            # sign of the -1 error sentinel.
            self.lib.NET_DVR_RealPlay_V40.restype = ctypes.c_int
            
            self.lib.NET_DVR_StopRealPlay.argtypes = [ctypes.c_int]
            self.lib.NET_DVR_StopRealPlay.restype = ctypes.c_bool

            self.lib.NET_DVR_SetRealDataCallBackEx.argtypes = [
                ctypes.c_int, fRealDataCallBack_V30, ctypes.c_void_p
            ]
            self.lib.NET_DVR_SetRealDataCallBackEx.restype = ctypes.c_bool

            # NET_DVR_GetLastError → returns DWORD (uint32)
            self.lib.NET_DVR_GetLastError.argtypes = []
            self.lib.NET_DVR_GetLastError.restype = ctypes.c_uint

            self.lib.NET_DVR_SetConnectTime = getattr(self.lib, "NET_DVR_SetConnectTime", None)
            if self.lib.NET_DVR_SetConnectTime:
                self.lib.NET_DVR_SetConnectTime.argtypes = [ctypes.c_uint, ctypes.c_uint]
                self.lib.NET_DVR_SetConnectTime.restype = ctypes.c_bool

            # Ensure lib_dir is in LD_LIBRARY_PATH for runtime dependencies
            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir not in ld_path:
                os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + ld_path

            # HCNetSDK is process-global. Configure/initialize it exactly once, and
            # do not let stopping one camera invalidate every other Hik stream.
            global _sdk_initialized, _sdk_instance_count
            with _sdk_lock:
                if not _sdk_initialized:
                    self.lib.NET_DVR_SetSDKInitCfg = getattr(self.lib, "NET_DVR_SetSDKInitCfg", None)
                    if self.lib.NET_DVR_SetSDKInitCfg:
                        self.lib.NET_DVR_SetSDKInitCfg.argtypes = [ctypes.c_uint, ctypes.c_void_p]
                        self.lib.NET_DVR_SetSDKInitCfg.restype = ctypes.c_bool
                        sdk_path = NET_DVR_LOCAL_SDK_PATH()
                        sdk_path.sPath = lib_dir.encode("utf-8")
                        if self.lib.NET_DVR_SetSDKInitCfg(2, ctypes.byref(sdk_path)):
                            logger.info("NET_DVR_SetSDKInitCfg type 2 (SDK path): %s", lib_dir)
                        else:
                            logger.warning(
                                "NET_DVR_SetSDKInitCfg type 2 was not applied (error %d); "
                                "continuing with the library's component search path",
                                self.lib.NET_DVR_GetLastError(),
                            )

                    if not self.lib.NET_DVR_Init():
                        raise RuntimeError(
                            f"NET_DVR_Init failed: {self.lib.NET_DVR_GetLastError()}"
                        )
                    _sdk_initialized = True
                    if self.lib.NET_DVR_SetConnectTime:
                        self.lib.NET_DVR_SetConnectTime(10000, 2)
                    get_version = getattr(self.lib, "NET_DVR_GetSDKVersion", None)
                    if get_version:
                        get_version.restype = ctypes.c_uint
                        logger.info("Hikvision HCNetSDK version word: 0x%08x", get_version())

                _sdk_instance_count += 1
                self._owns_sdk_reference = True
                
        except Exception as e:
            logger.warning(f"Could not load Hikvision SDK library: {e}. SDK wrapper will run in Simulation Mode.")
            self.lib = None

    def login(self, ip, port, username, password):
        if self.lib is None:
            logger.info(f"[Mock login] Logging into Hikvision camera at {ip}:{port}")
            self.logged_in = True
            self.lUserID = 9999
            self.start_chan = 1
            return True
            
        device_info = NET_DVR_DEVICEINFO_V30()
        
        # DDNS/WAN cameras occasionally return transient transport errors even
        # though the port is reachable. Retry network/busy failures, but never
        # retry credential, permission, version, or parameter errors.
        retryable_errors = {5, 7, 8, 9, 10, 11, 14, 24, 28, 29, 36}
        user_id = -1
        err = 0
        for attempt in range(1, 4):
            user_id = self.lib.NET_DVR_Login_V30(
                ip.encode("utf-8"),
                port,
                username.encode("utf-8"),
                password.encode("utf-8"),
                ctypes.byref(device_info),
            )
            if user_id >= 0:
                break
            err = self.lib.NET_DVR_GetLastError()
            if err not in retryable_errors or attempt == 3:
                break
            logger.warning(
                "Hikvision login attempt %d failed with transient error %d (%s); retrying",
                attempt,
                err,
                _error_text(err),
            )
            time.sleep(float(os.environ.get("HIK_LOGIN_RETRY_DELAY", "1")))
        
        if user_id >= 0:
            self.lUserID = user_id
            self.logged_in = True
            # DVR analog channels and NVR/IP channels use different SDK number
            # ranges. For an NVR, logical channel 1 normally maps to SDK channel
            # byStartDChan (commonly 33), not byStartChan.
            analog_count = int(device_info.byChanNum)
            digital_count = int(device_info.byIPChanNum) + (
                int(device_info.byHighDChanNum) << 8
            )
            if analog_count > 0:
                self.start_chan = int(device_info.byStartChan) or 1
                self.channel_count = analog_count
                self.is_digital_channel = False
            elif digital_count > 0:
                self.start_chan = int(device_info.byStartDChan) or 33
                self.channel_count = digital_count
                self.is_digital_channel = True
            else:
                # Some older devices omit counts but still accept channel 1.
                self.start_chan = int(device_info.byStartChan) or 1
                self.channel_count = 0
                self.is_digital_channel = False
            logger.info(
                "Hikvision login successful. User ID: %d, startChan: %d, "
                "channels: %d (%s)",
                self.lUserID,
                self.start_chan,
                self.channel_count,
                "digital" if self.is_digital_channel else "analog/direct",
            )
            return True
        else:
            logger.error("Hikvision login failed: %d (%s)", err, _error_text(err))
            return False

    def start_realplay(self, channel, callback_fn):
        """
        Starts preview and registers callback_fn to receive raw stream data.
        """
        if self.lib is None:
            logger.info("[Mock play] Starting mock Hikvision stream thread.")
            self.mock_running = True
            self.mock_thread = threading.Thread(
                target=self._run_mock_stream, 
                args=(callback_fn,),
                name="HikMockCallbackThread",
                daemon=True
            )
            self.mock_thread.start()
            self.play_handle = 8888
            return True
            
        preview_info = NET_DVR_PREVIEWINFO()
        preview_info.lChannel = channel
        preview_info.dwStreamType = 0  # Main stream
        preview_info.dwLinkMode = 0    # TCP
        preview_info.hPlayWnd = 0
        preview_info.bBlocked = True   # Must be True for data-only callback (no window)
        preview_info.dwDisplayBufNum = 1
        
        # Define C callback
        def c_callback(lPlayHandle, dwDataType, pBuffer, dwBufSize, pUser):
            # dwDataType == 2 is NET_DVR_STREAMDATA
            # In some cases we want all stream bytes
            if dwBufSize > 0 and pBuffer:
                data = ctypes.string_at(pBuffer, dwBufSize)
                callback_fn(data)
                
        self.callback_ref = fRealDataCallBack_V30(c_callback)
        
        # Follow the Linux demo bundled with this exact SDK: start preview first,
        # then register the data callback. Some device generations return a valid
        # handle but never invoke a callback passed directly to RealPlay_V40.
        self.play_handle = self.lib.NET_DVR_RealPlay_V40(
            self.lUserID,
            ctypes.byref(preview_info),
            None,
            None
        )
        
        # The SDK returns -1 on failure; any other value is a valid handle.
        if self.play_handle != -1:
            if not self.lib.NET_DVR_SetRealDataCallBackEx(
                self.play_handle, self.callback_ref, None
            ):
                err = self.lib.NET_DVR_GetLastError()
                self.lib.NET_DVR_StopRealPlay(self.play_handle)
                self.play_handle = -1
                logger.error(
                    "NET_DVR_SetRealDataCallBackEx failed: %d (%s)",
                    err,
                    _error_text(err),
                )
                return False
            logger.info(f"Hikvision RealPlay started. Handle: {self.play_handle}")
            return True
        else:
            err = self.lib.NET_DVR_GetLastError()
            logger.error(
                "NET_DVR_RealPlay_V40 failed: %d (%s)", err, _error_text(err)
            )
            return False

    def stop_realplay(self):
        if self.lib is None:
            logger.info("[Mock play] Stopping mock Hikvision stream thread.")
            self.mock_running = False
            if self.mock_thread:
                self.mock_thread.join(timeout=1.0)
            return True
            
        if self.play_handle != -1:
            self.lib.NET_DVR_StopRealPlay(self.play_handle)
            self.play_handle = -1
            self.callback_ref = None
            logger.info("Hikvision RealPlay stopped.")
            return True
        return False

    def logout(self):
        if self.lib is None:
            self.logged_in = False
            return True
            
        if self.logged_in and self.lUserID >= 0:
            self.lib.NET_DVR_Logout(self.lUserID)
            self.lUserID = -1
            self.logged_in = False
            logger.info("Hikvision logged out.")
            return True
        return False

    def cleanup(self):
        global _sdk_initialized, _sdk_instance_count
        if not self.lib or not self._owns_sdk_reference:
            return
        with _sdk_lock:
            self._owns_sdk_reference = False
            _sdk_instance_count = max(0, _sdk_instance_count - 1)
            if _sdk_instance_count == 0 and _sdk_initialized:
                self.lib.NET_DVR_Cleanup()
                _sdk_initialized = False
                logger.info("Hikvision SDK cleaned up.")

    def _run_mock_stream(self, callback_fn):
        """Simulates receiving raw H.264 packets at 25 FPS."""
        frame_idx = 0
        while self.mock_running:
            payload = f"hik_h264_packet_{frame_idx}_{time.time_ns()}".encode('utf-8')
            mock_h264 = b'\x00\x00\x00\x01\x27' + payload
            
            try:
                callback_fn(mock_h264)
            except Exception as e:
                logger.error(f"Error in Hik callback execution: {e}")
                
            frame_idx += 1
            time.sleep(1.0 / 25.0)
            
