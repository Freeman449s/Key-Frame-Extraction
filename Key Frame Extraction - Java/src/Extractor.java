import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.*;
import org.opencv.imgproc.*;

public class Extractor implements AutoCloseable {
    private String FILE_PATH;
    private int FPS;
    private VideoCapture cap;
    private String WINDOW_NAME;
    private int WIDTH;
    private int HEIGHT;
    private static double SHOT_BOUNDARY_THRESHOLD = 2E-7;
    private String KEY_FRAMES_FOLDER;
    private int N_FRAME;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        ArrayList<Integer> boundaryFrameIDs = null;
        ArrayList<double[]> momentInvarVecs = null;
        Mat frame = new Mat();
        boolean successful;
        try (Extractor extractor = new Extractor("Video.mp4", "Extractor", "./Key Frames/");) {
            momentInvarVecs = extractor.computeMomentInvarVecs();
            boundaryFrameIDs = extractor.shotBoundaryDetection(momentInvarVecs);
            //显示提取出的所有镜头边缘
            for (int i = 0; i <= boundaryFrameIDs.size() - 1; i++) {
                int frameID = boundaryFrameIDs.get(i);
                extractor.cap.set(CVConsts.CAP_PROP_POS_FRAMES, frameID);
                successful = extractor.cap.read(frame);
                if (successful) Imgcodecs.imwrite("Results/Boundaries/" + frameID + ".jpg", frame);
            }
            //选取两个边界帧之间的中间帧输出，用于评估镜头检测结果
            for (int i = 0; i <= boundaryFrameIDs.size() - 2; i++) {
                int frameID = (boundaryFrameIDs.get(i) + boundaryFrameIDs.get(i + 1)) / 2;
                extractor.cap.set(CVConsts.CAP_PROP_POS_FRAMES, frameID);
                successful = extractor.cap.read(frame);
                if (successful) Imgcodecs.imwrite("Results/Boundary Centers/" + frameID + ".jpg", frame);
            }
        }
        catch (Exception ex) {
            System.out.println("Error occured when trying to detect shot boundaries. Error message:");
            System.out.println(ex);
            in.nextLine();
            return;
        }

    }

    public Extractor(String filePath, String windowName, String keyFramesFolder) {
        FILE_PATH = filePath;
        cap = new VideoCapture(FILE_PATH);
        FPS = (int) cap.get(CVConsts.CAP_PROP_FPS);
        WINDOW_NAME = windowName;
        WIDTH = (int) cap.get(CVConsts.CAP_PROP_FRAME_WIDTH);
        HEIGHT = (int) cap.get(CVConsts.CAP_PROP_FRAME_HEIGHT);
        KEY_FRAMES_FOLDER = keyFramesFolder;
        N_FRAME = (int) cap.get(CVConsts.CAP_PROP_FRAME_COUNT);
    }

    public void close() {
        cap.release();
    }

    public ArrayList<Integer> shotBoundaryDetection(ArrayList<double[]> vecs) throws IOException, InterruptedException, TimeOutException {
        ArrayList<Integer> boundaryFrameIDs = new ArrayList<>();
        boundaryFrameIDs.add(0);
        int lastBoundaryID = 0;
        double[] vecOfLast = null;
        double[] vecOfThis = null;
        for (int i = 1; i <= N_FRAME - 1; i++) {
            vecOfLast = vecs.get(i - 1);
            vecOfThis = vecs.get(i);
            double momentInvarDist = computeMomentInvarDist(vecOfLast, vecOfThis);
            if (momentInvarDist > SHOT_BOUNDARY_THRESHOLD && i - lastBoundaryID > FPS / 2) {
                boundaryFrameIDs.add(i);
                lastBoundaryID = i;
            }
        }
        return boundaryFrameIDs;
        /*Mat frame = new Mat();
        Mat gray = new Mat();
        Mat lastGray = new Mat();
        //取第一帧作为边界帧，记录其矩不变向量
        boolean successful = cap.read(frame);
        frameID++;
        if (successful) {
            Imgproc.cvtColor(frame, lastGray, Imgproc.COLOR_BGR2GRAY);
            vecOfLast = computeMomentInvarVec(lastGray, frameID);
            successful = cap.read(frame);
            frameID++;
        }
        else {
            throw new IOException("Unable to read the first frame.");
        }
        while (successful) {
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
            double momentInvarDist;
            try {
                vecOfThis = computeMomentInvarVec(gray, frameID);
                momentInvarDist = computeMomentInvarDist(vecOfLast, vecOfThis);
                vecOfLast = vecOfThis;
                lastGray = gray;
            }
            catch (Exception ex) { //捕获到InterruptedException或TimeOutException时放弃此帧
                continue;
            }
            if (momentInvarDist > SHOT_BOUNDARY_THRESHOLD && frameID - lastBoundaryID > FPS / 2) {
                boundaryFrameIDs.add(frameID);
                lastBoundaryID = frameID;
            }
            successful = cap.read(frame);
            frameID++;
        }*/
    }

    private boolean isDarkFrame(Mat gray) {
        long sum = 0;
        for (int y = 0; y <= HEIGHT - 1; y++) {
            for (int x = 0; x <= WIDTH - 1; x++) {
                sum += gray.get(y, x)[0];
            }
        }
        return sum <= 255 / 4 * WIDTH * HEIGHT;
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

    //计算每一帧的矩不变向量
    private ArrayList<double[]> computeMomentInvarVecs() throws InterruptedException, TimeOutException {
        ArrayList<double[]> vecs = new ArrayList<>();
        int frameID = 0;
        Mat frame = new Mat();
        Mat gray = new Mat();
        boolean successful = cap.read(frame);
        while (successful) {
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
            vecs.add(computeMomentInvarVec(gray, frameID));
            successful = cap.read(frame);
            frameID++;
        }
        return vecs;
    }

    //计算矩不变向量
    private double[] computeMomentInvarVec(Mat gray, int frameID) throws InterruptedException, TimeOutException {
        System.out.println("进入computeMomentInvarVec，当前帧：" + frameID);
        long startTime = System.currentTimeMillis();
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
        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;
        System.out.println("\tcomputeMomentInvarVec退出，耗时" + duration + "ms.");
        return vec;
    }

    //计算两灰度图的矩不变向量的欧拉距离
    private double computeMomentInvarDist(double[] momentInvarVec_A, double[] momentInvarVec_B) throws InterruptedException, TimeOutException {
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