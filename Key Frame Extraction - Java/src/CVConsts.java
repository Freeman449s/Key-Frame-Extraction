public class CVConsts {
    //用于VideoCapture.get()的常量
    public static final int CAP_PROP_POS_MSEC = 0; //视频文件的当前位置（以毫秒为单位）或视频捕获时间戳
    public static final int CAP_PROP_POS_FRAMES = 1; //基于0的索引将被解码/捕获下一帧
    public static final int CAP_PROP_POS_AVI_RATIO = 2; //视频文件的相对位置：0-视频的开始，1-视频的结束
    public static final int CAP_PROP_FRAME_WIDTH = 3; //帧的宽度
    public static final int CAP_PROP_FRAME_HEIGHT = 4; //帧的高度
    public static final int CAP_PROP_FPS = 5; //帧速
    public static final int CAP_PROP_FOURCC = 6; //4个字符表示的视频编码器格式
    public static final int CAP_PROP_FRAME_COUNT = 7; //帧数
    public static final int CAP_PROP_FORMAT = 8; //byretrieve()返回的Mat对象的格式
    public static final int CAP_PROP_MODE = 9; //指示当前捕获模式的后端特定值
    public static final int CAP_PROP_BRIGHTNESS = 10; //图像的亮度（仅适用于相机）
    public static final int CAP_PROP_CONTRAST = 11; //图像对比度（仅适用于相机）
    public static final int CAP_PROP_SATURATION = 12; //图像的饱和度（仅适用于相机）
    public static final int CAP_PROP_HUE = 13; //图像的色相（仅适用于相机）
    public static final int CAP_PROP_GAIN = 14; //图像的增益（仅适用于相机）
    public static final int CAP_PROP_EXPOSURE = 15; //曝光（仅适用于相机）
    public static final int CAP_PROP_CONVERT_RGB = 16; //表示图像是否应转换为RGB的布尔标志
    public static final int CAP_PROP_WHITE_BALANCE = 17; //目前不支持
    public static final int CAP_PROP_RECTIFICATION = 18; //立体摄像机的整流标志
}
