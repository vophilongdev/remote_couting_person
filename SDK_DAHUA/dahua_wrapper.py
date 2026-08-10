import ctypes
import os
import time
import threading
import logging

logger = logging.getLogger("DahuaWrapper")

# Callback types
# void CALLBACK RealPlayCallBack(LLONG lRealPlayHandle, DWORD dwDataType, BYTE *pBuffer, DWORD dwBufSize, LDWORD dwUser)
fRealPlayCallBack = ctypes.CFUNCTYPE(
    None,
    ctypes.c_longlong,  # lRealPlayHandle
    ctypes.c_uint,      # dwDataType
    ctypes.POINTER(ctypes.c_ubyte),  # pBuffer
    ctypes.c_uint,      # dwBufSize
    ctypes.c_longlong   # dwUser
)

# void CALLBACK fRealDataCallBackEx(LLONG, DWORD, BYTE*, DWORD, LONG, LDWORD)
fRealDataCallBackEx = ctypes.CFUNCTYPE(
    None,
    ctypes.c_longlong,
    ctypes.c_uint,
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_uint,
    ctypes.c_long,
    ctypes.c_longlong,
)

fOriginalRealDataCallBack = ctypes.CFUNCTYPE(
    None,
    ctypes.c_longlong,
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_uint,
    ctypes.c_longlong,
)

_dahua_lock = threading.RLock()
_dahua_instance_count = 0
_dahua_initialized = False

# Structs
class NET_DEVICEINFO_Ex(ctypes.Structure):
    _fields_ = [
        ("sSerialNumber", ctypes.c_char * 48),
        ("byAlarmInPortNum", ctypes.c_byte),
        ("byAlarmOutPortNum", ctypes.c_byte),
        ("byDiskNum", ctypes.c_byte),
        ("byDVRType", ctypes.c_byte),
        ("byChanNum", ctypes.c_byte),
        # simplified for brevity
        ("reserved", ctypes.c_char * 500)
    ]

class DahuaSDK:
    def __init__(self, lib_path=None):
        self.lib = None
        self.logged_in = False
        self.lLoginID = 0
        self.play_handle = 0
        self.callback_ref = None
        self.mock_thread = None
        self.mock_running = False
        self._owns_sdk_reference = False
        
        if lib_path is None:
            # Look for libdhnetsdk.so in the expected workspace directory
            sdk_root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "General_NetSDK_ChnEng_JAVA_Linux64_IS_V3.052.0000001.0.R.200407",
                "General_NetSDK_ChnEng_JAVA_Linux64_IS_V3.052.0000001.0.R.200407",
            )
            lib_path = os.path.join(sdk_root, "libs", "linux64", "libdhnetsdk.so")
            
        try:
            # Set library loading paths for dependencies
            lib_dir = os.path.dirname(lib_path)
            # Load with RTLD_GLOBAL to resolve dependencies
            self.lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            logger.info("Successfully loaded Dahua NetSDK library.")
            
            # Setup signatures
            self.lib.CLIENT_Init.argtypes = [ctypes.c_void_p, ctypes.c_longlong]
            self.lib.CLIENT_Init.restype = ctypes.c_bool
            
            self.lib.CLIENT_Cleanup.argtypes = []
            self.lib.CLIENT_Cleanup.restype = None

            # CLIENT_GetLastError → returns DWORD
            self.lib.CLIENT_GetLastError.argtypes = []
            self.lib.CLIENT_GetLastError.restype = ctypes.c_uint
            
            self.lib.CLIENT_LoginEx2.argtypes = [
                ctypes.c_char_p, ctypes.c_ushort, ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(NET_DEVICEINFO_Ex), ctypes.POINTER(ctypes.c_int)
            ]
            self.lib.CLIENT_LoginEx2.restype = ctypes.c_longlong
            
            self.lib.CLIENT_Logout.argtypes = [ctypes.c_longlong]
            self.lib.CLIENT_Logout.restype = ctypes.c_bool
            
            self.lib.CLIENT_RealPlayEx.argtypes = [ctypes.c_longlong, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
            self.lib.CLIENT_RealPlayEx.restype = ctypes.c_longlong

            self.lib.CLIENT_SetRealDataCallBackEx.argtypes = [
                ctypes.c_longlong,
                fRealDataCallBackEx,
                ctypes.c_longlong,
                ctypes.c_uint,
            ]
            self.lib.CLIENT_SetRealDataCallBackEx.restype = ctypes.c_bool

            self.lib.CLIENT_SetOriginalRealDataCallBack.argtypes = [
                ctypes.c_longlong,
                fOriginalRealDataCallBack,
                ctypes.c_longlong,
            ]
            self.lib.CLIENT_SetOriginalRealDataCallBack.restype = ctypes.c_bool
            
            self.lib.CLIENT_StopRealPlayEx.argtypes = [ctypes.c_longlong]
            self.lib.CLIENT_StopRealPlayEx.restype = ctypes.c_bool

            self.lib.CLIENT_SetConnectTime = getattr(self.lib, "CLIENT_SetConnectTime", None)
            if self.lib.CLIENT_SetConnectTime:
                self.lib.CLIENT_SetConnectTime.argtypes = [ctypes.c_int, ctypes.c_int]
                self.lib.CLIENT_SetConnectTime.restype = None
            
            # Initialize SDK exactly once per process
            global _dahua_initialized, _dahua_instance_count
            with _dahua_lock:
                if not _dahua_initialized:
                    self.lib.CLIENT_Init(None, 0)
                    if self.lib.CLIENT_SetConnectTime:
                        self.lib.CLIENT_SetConnectTime(300000, 1)
                        logger.info("Set Dahua connection timeout to 300,000 ms (5 minutes)")
                    _dahua_initialized = True
                _dahua_instance_count += 1
                self._owns_sdk_reference = True
        except Exception as e:
            logger.warning(f"Could not load Dahua SDK library: {e}. SDK wrapper will run in Simulation Mode.")
            self.lib = None

    def login(self, ip, port, username, password):
        if self.lib is None:
            logger.info(f"[Mock login] Logging into Dahua camera at {ip}:{port}")
            self.logged_in = True
            self.lLoginID = 9999
            return True
            
        device_info = NET_DEVICEINFO_Ex()
        error_code = ctypes.c_int(0)
        
        login_id = self.lib.CLIENT_LoginEx2(
            ip.encode('utf-8'),
            port,
            username.encode('utf-8'),
            password.encode('utf-8'),
            0, # TCP
            None,
            ctypes.byref(device_info),
            ctypes.byref(error_code)
        )
        
        if login_id != 0:
            self.lLoginID = login_id
            self.logged_in = True
            logger.info(f"Dahua login successful. ID: {self.lLoginID}")
            return True
        else:
            logger.error(f"Dahua login failed. Error code: {error_code.value}")
            return False

    def start_realplay(self, channel, callback_fn):
        """
        Starts the realplay and routes raw H.264 data to callback_fn.
        callback_fn signature: callback_fn(data_bytes)
        """
        if self.lib is None:
            logger.info("[Mock play] Starting mock Dahua stream thread.")
            self.mock_running = True
            self.mock_thread = threading.Thread(
                target=self._run_mock_stream, 
                args=(callback_fn,), 
                name="DahuaMockCallbackThread",
                daemon=True
            )
            self.mock_thread.start()
            self.play_handle = 8888
            return True
            
        # Define the C callback wrapper
        def c_callback(lRealPlayHandle, pBuffer, dwBufSize, dwUser):
            # Original encoded elementary stream, copied before the SDK releases it.
            if dwBufSize > 0 and pBuffer:
                data = ctypes.string_at(pBuffer, dwBufSize)
                callback_fn(data)
                
        # Keep a reference to prevent garbage collection
        self.callback_ref = fOriginalRealDataCallBack(c_callback)
        
        # Dahua SDK channels are 0-based.  The camera API and config use 1-based
        # numbering, so subtract 1 here.  A value ≤ 0 from the API means "first
        # channel".
        sdk_channel = max(channel - 1, 0)
        logger.info(
            "Dahua RealPlayEx: loginID=%s, sdk_channel=%d (config channel=%d)",
            self.lLoginID, sdk_channel, channel,
        )
        # Set callback and play
        self.play_handle = self.lib.CLIENT_RealPlayEx(self.lLoginID, sdk_channel, None, 0)
        if self.play_handle != 0:
            callback_ok = self.lib.CLIENT_SetOriginalRealDataCallBack(
                self.play_handle, self.callback_ref, 0
            )
            if callback_ok:
                logger.info(f"Dahua RealPlay started with callback. Handle: {self.play_handle}")
                return True
            else:
                error_code = self.lib.CLIENT_GetLastError()
                logger.error(
                    "CLIENT_SetOriginalRealDataCallBack failed. Error code: %s",
                    error_code,
                )
                self.lib.CLIENT_StopRealPlayEx(self.play_handle)
                self.play_handle = 0
                return False
        else:
            error_code = self.lib.CLIENT_GetLastError()
            logger.error("CLIENT_RealPlayEx failed. Error code: %s", error_code)
            return False

    def stop_realplay(self):
        if self.lib is None:
            logger.info("[Mock play] Stopping mock Dahua stream thread.")
            self.mock_running = False
            if self.mock_thread:
                self.mock_thread.join(timeout=1.0)
            return True
            
        if self.play_handle != 0:
            self.lib.CLIENT_StopRealPlayEx(self.play_handle)
            self.play_handle = 0
            self.callback_ref = None
            logger.info("Dahua RealPlay stopped.")
            return True
        return False

    def logout(self):
        if self.lib is None:
            self.logged_in = False
            return True
            
        if self.logged_in and self.lLoginID != 0:
            self.lib.CLIENT_Logout(self.lLoginID)
            self.lLoginID = 0
            self.logged_in = False
            logger.info("Dahua logged out.")
            return True
        return False

    def cleanup(self):
        global _dahua_initialized, _dahua_instance_count
        if not self.lib or not self._owns_sdk_reference:
            return
        with _dahua_lock:
            self._owns_sdk_reference = False
            _dahua_instance_count = max(0, _dahua_instance_count - 1)
            if _dahua_instance_count == 0 and _dahua_initialized:
                self.lib.CLIENT_Cleanup()
                _dahua_initialized = False
                logger.info("Dahua SDK cleaned up.")

    def _run_mock_stream(self, callback_fn):
        """Simulates receiving raw H.264 packets at 25 FPS."""
        frame_idx = 0
        while self.mock_running:
            # Generate a mock packet payload
            # We put some markers to identify it
            payload = f"dahua_h264_packet_{frame_idx}_{time.time_ns()}".encode('utf-8')
            # Add a mock start code or H.264 structure
            mock_h264 = b'\x00\x00\x00\x01\x27' + payload # Mock SPS/NAL unit
            
            try:
                # Call the callback immediately (simulating Async Callback Thread)
                callback_fn(mock_h264)
            except Exception as e:
                logger.error(f"Error in Dahua callback execution: {e}")
                
            frame_idx += 1
            time.sleep(1.0 / 25.0) # 25 FPS
