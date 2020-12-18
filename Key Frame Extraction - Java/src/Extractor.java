import java.math.*;
import java.util.concurrent.*;

import org.opencv.core.*;
import org.opencv.video.*;
import org.opencv.videoio.*;
import org.opencv.highgui.*;
import org.opencv.imgproc.*;

//todo: 已实现矩不变量向量距离的计算，等待调试
//todo: 等待实现基于矩不变量的镜头分割

public class Extractor {
    private String FILE_PATH;
    private int FPS;
    private VideoCapture cap;
    private String WINDOW_NAME;
    private int WIDTH;
    private int HEIGHT;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        Extractor extractor = new Extractor("Video.mp4");
        Mat frame = new Mat();
        Mat gray = new Mat();
        boolean successful = extractor.cap.read(frame);
        while (successful) {
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
            HighGui.imshow(extractor.WINDOW_NAME, gray);
            HighGui.waitKey(1000 / extractor.FPS);
            successful = extractor.cap.read(frame);
        }
    }

    public Extractor(String filePath, String windowName) {
        FILE_PATH = filePath;
        cap = new VideoCapture(FILE_PATH);
        FPS = (int) cap.get(CVConsts.CAP_PROP_FPS);
        WINDOW_NAME = windowName;
        WIDTH = (int) cap.get(CVConsts.CAP_PROP_FRAME_WIDTH);
        HEIGHT = (int) cap.get(CVConsts.CAP_PROP_FRAME_HEIGHT);
    }

    public Extractor(String filePath) {
        this(filePath, "Extractor");
    }

    //计算图像矩
    private double computeMoment(Mat gray, double p, double q) {
        double m = 0;
        for (int y = 0; y <= HEIGHT - 1; y++) { //Mat的行数等于图像的高度
            for (int x = 0; x <= WIDTH - 1; x++) {
                m += Math.pow(x, p) * Math.pow(y, q) * gray.get(y, x)[0];
            }
        }
        return m;
    }

    //计算灰度图的重心
    //参数
    //sum: 灰度图中所有像素像素值的和
    private double[] computeMassCenter(Mat gray, double sum) {
        double xBar = 0, yBar = 0;
        for (int y = 0; y <= HEIGHT - 1; y++) {
            for (int x = 0; x <= WIDTH - 1; x++) {
                xBar += x * gray.get(y, x)[0];
                yBar += y * gray.get(y, x)[0];
            }
        }
        xBar /= sum;
        yBar /= sum;
        double[] ret = {xBar, yBar};
        return ret;
    }

    //计算矩不变量
    //参数
    //sum: 灰度图中所有像素像素值的和
    //xBar: 重心的横坐标
    //yBar: 中心的纵坐标
    private double computeMomentInvariant(Mat gray, double p, double q, double xBar, double yBar, double sum) {
        double n = 0;
        for (int y = 0; y <= HEIGHT - 1; y++) {
            for (int x = 0; x <= WIDTH - 1; x++) {
                n += Math.pow(x - xBar, p) * Math.pow(y - yBar, q) * gray.get(y, x)[0];
            }
        }
        double normFactor = Math.pow(sum, 1 + (p + q) / 2); //归一化因子
        n /= normFactor;
        return n;
    }

    //计算矩不变向量
    private double[] computeMomentInvarVec(Mat gray) throws InterruptedException, TimeOutException {
        double sum = computeMoment(gray, 0, 0);
        double[] massCenter = computeMassCenter(gray, sum);
        double xBar = massCenter[0];
        double yBar = massCenter[1];
        double[] invariants = new double[7];
        //并行计算矩不变量
        ExecutorService executor = Executors.newCachedThreadPool();
        executor.execute(new MomentInvarComputeTask(gray, 0, 2, xBar, yBar, sum, invariants, 0));
        executor.execute(new MomentInvarComputeTask(gray, 0, 3, xBar, yBar, sum, invariants, 1));
        executor.execute(new MomentInvarComputeTask(gray, 1, 1, xBar, yBar, sum, invariants, 2));
        executor.execute(new MomentInvarComputeTask(gray, 1, 2, xBar, yBar, sum, invariants, 3));
        executor.execute(new MomentInvarComputeTask(gray, 2, 0, xBar, yBar, sum, invariants, 4));
        executor.execute(new MomentInvarComputeTask(gray, 2, 1, xBar, yBar, sum, invariants, 5));
        executor.execute(new MomentInvarComputeTask(gray, 3, 0, xBar, yBar, sum, invariants, 6));
        executor.shutdown();
        boolean successful = executor.awaitTermination(600, TimeUnit.SECONDS);
        if (!successful)
            throw new TimeOutException("Task \"Compute Moment Invariants\" didn't finish in time."); //太长时间没能计算完毕，抛出异常告知

        double n20 = invariants[4];
        double n02 = invariants[0];
        double n11 = invariants[2];
        double n30 = invariants[6];
        double n12 = invariants[3];
        double n21 = invariants[5];
        double n03 = invariants[1];
        double phi1 = n20 + n02;
        double phi2 = Math.pow(n20 - n02, 2) + 4 * Math.pow(n11, 2);
        double phi3 = Math.pow(n30 - 3 * n12, 2) + Math.pow(3 * n21 - n03, 2);
        double[] vec = {phi1, phi2, phi3};
        return vec;
    }

    //计算两灰度图的矩不变向量的欧拉距离
    private double computeMomentInvarDist(Mat grayA, Mat grayB) throws InterruptedException, TimeOutException {
        double[] momentInvarVec_A = computeMomentInvarVec(grayA);
        double[] momentInvarVec_B = computeMomentInvarVec(grayB);
        double deltaX = momentInvarVec_A[0] - momentInvarVec_B[0];
        double deltaY = momentInvarVec_A[1] - momentInvarVec_B[1];
        double deltaZ = momentInvarVec_A[2] - momentInvarVec_B[2];
        return Math.pow(deltaX, 2) + Math.pow(deltaY, 2) + Math.pow(deltaZ, 2);
    }

    //用于并行计算矩不变量的内部类
    class MomentInvarComputeTask implements Runnable {
        private Mat gray;
        private double xBar;
        private double yBar;
        private double sum;
        private double p;
        private double q;
        private double[] result;
        private int index;

        MomentInvarComputeTask(Mat gray, double p, double q, double xBar, double yBar, double sum, double[] result, int index) {
            this.gray = gray;
            this.xBar = xBar;
            this.yBar = yBar;
            this.sum = sum;
            this.p = p;
            this.q = q;
            this.result = result;
            this.index = index; //在数组的哪个位置写入答案
        }

        public void run() {
            double n = 0;
            for (int y = 0; y <= HEIGHT - 1; y++) {
                for (int x = 0; x <= WIDTH - 1; x++) {
                    n += Math.pow(x - xBar, p) * Math.pow(y - yBar, q) * gray.get(y, x)[0];
                }
            }
            double normFactor = Math.pow(sum, 1 + (p + q) / 2); //归一化因子
            n /= normFactor;
            result[index] = n;
        }
    }
}

class TimeOutException extends Exception {
    public TimeOutException(String msg) {
        super(msg);
    }
}