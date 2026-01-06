"""
Translation module - provides multi-language text support for detection models
"""

class Translations:
    """Translation module, provides multi-language text support"""
    
    # Current language (default English)
    current_language = "en"
    
    # Multi-language translation dictionary
    translations = {
        # Application title and general
        "app_title": {
            "zh": "个性化老年健康监测",
            "en": "Personalized Geriatric Wellness Monitoring",
            "es": "Monitoreo de Bienestar Geriátrico Personalizado",
            "hi": "व्यक्तिगत वृद्ध कल्याण निगरानी"
        },
        "ready": {
            "zh": "准备就绪",
            "en": "Ready",
            "es": "Listo",
            "hi": "तैयार"
        },
        
        # Menu items
        "tools_menu": {
            "zh": "工具",
            "en": "Tools",
            "es": "Herramientas",
            "hi": "उपकरण"
        },
        "mode_menu": {
            "zh": "模式",
            "en": "Mode",
            "es": "Modo",
            "hi": "मोड"
        },
        "help_menu": {
            "zh": "帮助",
            "en": "Help",
            "es": "Ayuda",
            "hi": "मदद"
        },
        "language_menu": {
            "zh": "语言",
            "en": "Language",
            "es": "Idioma",
            "hi": "भाषा"
        },
        "chinese": {
            "zh": "中文",
            "en": "Chinese",
            "es": "Chino",
            "hi": "चीनी"
        },
        "english": {
            "zh": "英文",
            "en": "English",
            "es": "Inglés",
            "hi": "अंग्रेज़ी"
        },
        "spanish": {
            "zh": "西班牙语",
            "en": "Spanish",
            "es": "Español",
            "hi": "स्पेनिश"
        },
        "hindi": {
            "zh": "印地语",
            "en": "Hindi",
            "es": "Hindi",
            "hi": "हिंदी"
        },
        "skeleton_display": {
            "zh": "显示骨架",
            "en": "Show Skeleton",
            "es": "Mostrar esqueleto",
            "hi": "कंकाल दिखाएं"
        },
        "video_file": {
            "zh": "打开视频文件",
            "en": "Open Video File",
            "es": "Abrir archivo de video",
            "hi": "वीडियो फ़ाइल खोलें"
        },
        "camera_mode": {
            "zh": "切换到摄像头模式",
            "en": "Switch to Camera Mode",
            "es": "Cambiar a modo cámara",
            "hi": "कैमरा मोड पर स्विच करें"
        },
        "rotation_mode": {
            "zh": "竖屏模式",
            "en": "Vertical Mode",
            "es": "Modo vertical",
            "hi": "वर्टिकल मोड"
        },
        "detection_mode": {
            "zh": "检测模式",
            "en": "Detection Mode",
            "es": "Modo de detección",
            "hi": "डिटेक्शन मोड"
        },
        "stats_mode": {
            "zh": "统计管理模式",
            "en": "Statistics Mode",
            "es": "Modo de estadísticas",
            "hi": "आँकड़े मोड"
        },
        "about": {
            "zh": "关于",
            "en": "About",
            "es": "Acerca de",
            "hi": "के बारे में"
        },
        
        # Control panel
        "detection_data": {
            "zh": "检测数据",
            "en": "Detection Data",
            "es": "Datos de detección",
            "hi": "डिटेक्शन डेटा"
        },
        "detections_found": {
            "zh": "检测数量:",
            "en": "Detections:",
            "es": "Detecciones:",
            "hi": "डिटेक्शन:"
        },
        "fps": {
            "zh": "帧率:",
            "en": "FPS:",
            "es": "FPS:",
            "hi": "FPS:"
        },
        "control_options": {
            "zh": "控制选项",
            "en": "Control Options",
            "es": "Opciones de control",
            "hi": "नियंत्रण विकल्प"
        },
        "detection_type": {
            "zh": "检测类型:",
            "en": "Detection Type:",
            "es": "Tipo de detección:",
            "hi": "डिटेक्शन प्रकार:"
        },
        "camera": {
            "zh": "摄像头:",
            "en": "Camera:",
            "es": "Cámara:",
            "hi": "कैमरा:"
        },
        "reset": {
            "zh": "重置",
            "en": "Reset",
            "es": "Restablecer",
            "hi": "रीसेट करें"
        },
        "save_results": {
            "zh": "保存结果",
            "en": "Save Results",
            "es": "Guardar resultados",
            "hi": "परिणाम सहेजें"
        },
        "detection_status": {
            "zh": "检测状态",
            "en": "Detection Status",
            "es": "Estado de detección",
            "hi": "डिटेक्शन स्थिति"
        },
        "current_status": {
            "zh": "当前状态:",
            "en": "Current Status:",
            "es": "Estado actual:",
            "hi": "वर्तमान स्थिति:"
        },
        
        # Detection types
        "pose_detection": {
            "zh": "姿态检测",
            "en": "Pose Detection",
            "es": "Detección de pose",
            "hi": "पोज़ डिटेक्शन"
        },
        "object_detection": {
            "zh": "目标检测",
            "en": "Object Detection",
            "es": "Detección de objetos",
            "hi": "ऑब्जेक्ट डिटेक्शन"
        },
        "action_recognition": {
            "zh": "动作识别",
            "en": "Action Recognition",
            "es": "Reconocimiento de acciones",
            "hi": "एक्शन रिकग्निशन"
        },
        "custom_detection": {
            "zh": "自定义检测",
            "en": "Custom Detection",
            "es": "Detección personalizada",
            "hi": "कस्टम डिटेक्शन"
        },
        
        # Model types
        "model_type": {
            "zh": "模型选择:",
            "en": "Model:",
            "es": "Modelo:",
            "hi": "मॉडल:"
        },
        "lightweight": {
            "zh": "轻量级模型",
            "en": "Lightweight",
            "es": "Modelo ligero",
            "hi": "हल्का मॉडल"
        },
        "balanced": {
            "zh": "平衡模型",
            "en": "Balanced",
            "es": "Modelo equilibrado",
            "hi": "संतुलित मॉडल"
        },
        "performance": {
            "zh": "高性能模型",
            "en": "Performance",
            "es": "Modelo de alto rendimiento",
            "hi": "उच्च प्रदर्शन मॉडल"
        },
        "mirror_mode": {
            "zh": "镜像模式",
            "en": "Mirror Mode",
            "es": "Modo espejo",
            "hi": "दर्पण मोड"
        },
        
        # Status bar messages
        "welcome": {
            "zh": "欢迎使用个性化老年健康监测系统",
            "en": "Welcome to Personalized Geriatric Wellness Monitoring",
            "es": "Bienvenido al Monitoreo de Bienestar Geriátrico Personalizado",
            "hi": "व्यक्तिगत वृद्ध कल्याण निगरानी में आपका स्वागत है"
        },
        "language_changed": {
            "zh": "语言已更改",
            "en": "Language changed",
            "es": "Idioma cambiado",
            "hi": "भाषा बदली गई"
        },
        "switched_to_detection": {
            "zh": "已切换到检测模式",
            "en": "Switched to detection mode",
            "es": "Cambiado a modo de detección",
            "hi": "डिटेक्शन मोड पर स्विच किया गया"
        },
        "switched_to_stats": {
            "zh": "已切换到统计管理模式",
            "en": "Switched to statistics mode",
            "es": "Cambiado a modo de estadísticas",
            "hi": "आँकड़े मोड पर स्विच किया गया"
        },
        "detection_reset": {
            "zh": "检测已重置",
            "en": "Detection reset",
            "es": "Detección reiniciada",
            "hi": "डिटेक्शन रीसेट"
        },
        "results_saved": {
            "zh": "结果已保存",
            "en": "Results saved",
            "es": "Resultados guardados",
            "hi": "परिणाम सहेजे गए"
        },
        
        # About dialog
        "about_title": {
            "zh": "关于个性化老年健康监测系统",
            "en": "About Personalized Geriatric Wellness Monitoring",
            "es": "Acerca del Monitoreo de Bienestar Geriátrico Personalizado",
            "hi": "व्यक्तिगत वृद्ध कल्याण निगरानी के बारे में"
        },
        "about_content": {
            "zh": "个性化老年健康监测系统 v1.0\n\n基于AI技术的老年健康监测系统\n\n支持姿态检测、动作识别和健康状态分析",
            "en": "Personalized Geriatric Wellness Monitoring v1.0\n\nAI-powered geriatric health monitoring system\n\nSupports pose detection, action recognition, and health status analysis",
            "es": "Monitoreo de Bienestar Geriátrico Personalizado v1.0\n\nSistema de monitoreo de salud geriátrica basado en IA\n\nSoporta detección de pose, reconocimiento de acciones y análisis del estado de salud",
            "hi": "व्यक्तिगत वृद्ध कल्याण निगरानी v1.0\n\nAI-संचालित वृद्ध स्वास्थ्य निगरानी प्रणाली\n\nपोज़ डिटेक्शन, एक्शन रिकग्निशन और स्वास्थ्य स्थिति विश्लेषण का समर्थन करता है"
        },
        
        # Video related
        "open_video": {
            "zh": "打开视频文件",
            "en": "Open Video File",
            "es": "Abrir archivo de video",
            "hi": "वीडियो फ़ाइल खोलें"
        },
        "video_files": {
            "zh": "视频文件 (*.mp4 *.avi *.mov *.wmv *.mkv)",
            "en": "Video Files (*.mp4 *.avi *.mov *.wmv *.mkv)",
            "es": "Archivos de video (*.mp4 *.avi *.mov *.wmv *.mkv)",
            "hi": "वीडियो फ़ाइलें (*.mp4 *.avi *.mov *.wmv *.mkv)"
        },
        "error_opening_video": {
            "zh": "无法打开视频文件",
            "en": "Failed to open video file",
            "es": "No se pudo abrir el archivo de video",
            "hi": "वीडियो फ़ाइल खोलने में असफल"
        },
        "video_loaded": {
            "zh": "视频已加载: ",
            "en": "Video loaded: ",
            "es": "Video cargado: ",
            "hi": "वीडियो लोड किया गया: "
        },
        
        # Statistics panel
        "detection_statistics": {
            "zh": "检测统计",
            "en": "Detection Statistics",
            "es": "Estadísticas de detección",
            "hi": "डिटेक्शन आँकड़े"
        },
        "overview": {
            "zh": "概览",
            "en": "Overview",
            "es": "Resumen",
            "hi": "अवलोकन"
        },
        "daily_stats": {
            "zh": "每日统计",
            "en": "Daily Stats",
            "es": "Estadísticas diarias",
            "hi": "दैनिक आँकड़े"
        },
        "detection_types": {
            "zh": "检测类型",
            "en": "Detection Types",
            "es": "Tipos de detección",
            "hi": "डिटेक्शन प्रकार"
        },
        "export_data": {
            "zh": "导出数据",
            "en": "Export Data",
            "es": "Exportar datos",
            "hi": "डेटा निर्यात करें"
        },
        "overall_statistics": {
            "zh": "总体统计",
            "en": "Overall Statistics",
            "es": "Estadísticas generales",
            "hi": "समग्र आँकड़े"
        },
        "total_detections": {
            "zh": "总检测数",
            "en": "Total Detections",
            "es": "Total de detecciones",
            "hi": "कुल डिटेक्शन"
        },
        "detection_types_summary": {
            "zh": "检测类型汇总",
            "en": "Detection Types Summary",
            "es": "Resumen de tipos de detección",
            "hi": "डिटेक्शन प्रकार सारांश"
        },
        "count": {
            "zh": "数量",
            "en": "Count",
            "es": "Recuento",
            "hi": "गिनती"
        },
        "daily_statistics": {
            "zh": "每日统计",
            "en": "Daily Statistics",
            "es": "Estadísticas diarias",
            "hi": "दैनिक आँकड़े"
        },
        "date": {
            "zh": "日期",
            "en": "Date",
            "es": "Fecha",
            "hi": "तारीख"
        },
        "detection_types_breakdown": {
            "zh": "检测类型分解",
            "en": "Detection Types Breakdown",
            "es": "Desglose de tipos de detección",
            "hi": "डिटेक्शन प्रकार विवरण"
        },
        "total_count": {
            "zh": "总数量",
            "en": "Total Count",
            "es": "Recuento total",
            "hi": "कुल गिनती"
        },
        "daily_average": {
            "zh": "日均",
            "en": "Daily Average",
            "es": "Promedio diario",
            "hi": "दैनिक औसत"
        },
        "last_detection": {
            "zh": "最后检测",
            "en": "Last Detection",
            "es": "Última detección",
            "hi": "अंतिम डिटेक्शन"
        },
        "export_description": {
            "zh": "导出检测数据用于进一步分析或备份",
            "en": "Export detection data for further analysis or backup",
            "es": "Exportar datos de detección para análisis adicional o respaldo",
            "hi": "आगे के विश्लेषण या बैकअप के लिए डिटेक्शन डेटा निर्यात करें"
        },
        "export_json": {
            "zh": "导出JSON",
            "en": "Export JSON",
            "es": "Exportar JSON",
            "hi": "JSON निर्यात करें"
        },
        "export_csv": {
            "zh": "导出CSV",
            "en": "Export CSV",
            "es": "Exportar CSV",
            "hi": "CSV निर्यात करें"
        },
        "clear_data": {
            "zh": "清除数据",
            "en": "Clear Data",
            "es": "Limpiar datos",
            "hi": "डेटा साफ़ करें"
        },
        "save_file": {
            "zh": "保存文件",
            "en": "Save File",
            "es": "Guardar archivo",
            "hi": "फ़ाइल सहेजें"
        },
        "confirm_clear": {
            "zh": "确认清除",
            "en": "Confirm Clear",
            "es": "Confirmar limpieza",
            "hi": "साफ़ करने की पुष्टि करें"
        },
        "clear_data_warning": {
            "zh": "确定要清除所有数据吗？此操作不可撤销。",
            "en": "Are you sure you want to clear all data? This action cannot be undone.",
            "es": "¿Estás seguro de que quieres limpiar todos los datos? Esta acción no se puede deshacer.",
            "hi": "क्या आप वाकई सभी डेटा साफ़ करना चाहते हैं? यह क्रिया पूर्ववत नहीं की जा सकती।"
        },
        
        # Additional menu items
        "file_menu": {
            "zh": "文件",
            "en": "File",
            "es": "Archivo",
            "hi": "फ़ाइल"
        },
        "exit": {
            "zh": "退出",
            "en": "Exit",
            "es": "Salir",
            "hi": "बाहर निकलें"
        }
    }
    
    @classmethod
    def get(cls, key):
        """Get translation text for current language"""
        if key in cls.translations:
            return cls.translations[key][cls.current_language]
        return key
    
    @classmethod
    def set_language(cls, language):
        """Set current language"""
        if language in ["zh", "en", "es", "hi"]:
            cls.current_language = language
            return True
        return False
