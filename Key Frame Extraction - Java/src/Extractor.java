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
    private static double SHOT_BOUNDARY_THRESHOLD = 2E-7; //利用矩不变量法进行镜头边缘检测的阈值，推荐设为2E-7
    private static double KEY_FRAME_INVAR_THRESHOLD = 3E-6; //用于通过矩不变量法进行关键帧提取的阈值，推荐设为3E-6
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
        try (Extractor extractor = new Extractor("Video - Complete.mp4", "Extractor", "./Results/Key Frames/");) {
            momentInvarVecs = extractor.computeMomentInvarVecs();
            boundaryFrameIDs = extractor.shotBoundaryDetection(momentInvarVecs);
            extractor.keyFrameExtraction_Invar(boundaryFrameIDs, momentInvarVecs);
        }
        catch (Exception ex) {
            System.out.println("Error occured when trying to detect shot boundaries. Error message:");
            System.out.println(ex);
            in.nextLine();
            return;
        }

    }

    //构造函数
    //keyFramesFolder末尾应有分隔符"/"或"\\"
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

    //镜头边缘检测
    public ArrayList<Integer> shotBoundaryDetection(ArrayList<double[]> vecs) throws IOException, InterruptedException, TimeoutException {
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
        /*ArrayList<Integer> boundaryFrameIDs = new ArrayList<>();
        boundaryFrameIDs.add(0);
        Mat frame = new Mat();
        Mat gray = new Mat();
        int frameID = 0;
        int lastBoundaryID = 0;
        double[] vecOfLast;
        double[] vecOfThis;
        //取第一帧作为边界帧，记录其矩不变向量
        boolean successful = cap.read(frame);
        if (successful) {
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
            vecOfLast = computeMomentInvarVec(gray, frameID);
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
        }
        return boundaryFrameIDs;*/
    }

    //基于矩不变量提取关键帧
    public void keyFrameExtraction_Invar(ArrayList<Integer> boundaryIDs, ArrayList<double[]> vecs) throws InterruptedException, TimeoutException {
        System.out.print("Extracting key frames");
        int dotBoundary = boundaryIDs.size() / 5;
        int nDotsPrinted = 0;
        //各个镜头独立处理
        for (int i = 0; i <= boundaryIDs.size() - 2; i++) {
            inShotKeyFrameExtraction_Invar(vecs, boundaryIDs.get(i), boundaryIDs.get(i + 1));
            if (i >= dotBoundary) {
                System.out.print(".");
                dotBoundary += boundaryIDs.size() / 5;
                nDotsPrinted++;
            }
        }
        if (nDotsPrinted < 5) System.out.print(".");
        //单独处理最后一个镜头
        inShotKeyFrameExtraction_Invar(vecs, boundaryIDs.get(boundaryIDs.size() - 1), N_FRAME);
        System.out.println("OK");
    }

    //镜头内关键帧提取，基于矩不变量
    //直接在指定文件夹内写入关键帧，无返回值
    private void inShotKeyFrameExtraction_Invar(ArrayList<double[]> vecs, int left, int right) throws InterruptedException, TimeoutException {
        Mat frame = new Mat();
        Mat gray = new Mat();
        //第一帧作为关键帧
        int lastKeyFrameID = left;
        cap.set(CVConsts.CAP_PROP_POS_FRAMES, lastKeyFrameID);
        boolean successful = cap.read(frame);
        //如果读取第一帧不成功，则持续读帧，直到读帧成功或进入下一个镜头
        while (!successful) {
            lastKeyFrameID++;
            if (lastKeyFrameID >= right) break;
            successful = cap.read(frame);
        }
        if (lastKeyFrameID >= right) return; //当前镜头内无法读取任何一帧
        if (!isDarkFrame(frame)) Imgcodecs.imwrite(KEY_FRAMES_FOLDER + lastKeyFrameID + ".jpg", frame);
        double[] vecOfLastKeyFrame = vecs.get(lastKeyFrameID);
        //遍历镜头中的各帧
        for (int i = lastKeyFrameID + 1; i <= right - 1; i++) {
            double[] vecOfThis = vecs.get(i);
            double dist = computeMomentInvarDist(vecOfLastKeyFrame, vecOfThis);
            if (dist > KEY_FRAME_INVAR_THRESHOLD && i - lastKeyFrameID > FPS / 2) {
                cap.set(CVConsts.CAP_PROP_POS_FRAMES, i);
                successful = cap.read(frame);
                if (!successful) continue;
                Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
                if (!isDarkFrame(gray)) {
                    Imgcodecs.imwrite(KEY_FRAMES_FOLDER + i + ".jpg", frame);
                    lastKeyFrameID = i;
                    vecOfLastKeyFrame = vecOfThis;
                }
            }
        }
    }

    //todo: 自适应暗帧检测
    private boolean isDarkFrame(Mat gray) {
        long sum = 0;
        for (int y = 0; y <= HEIGHT - 1; y++) {
            for (int x = 0; x <= WIDTH - 1; x++) {
                sum += gray.get(y, x)[0];
            }
        }
        return sum <= 255 / 8 * WIDTH * HEIGHT;
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
    private ArrayList<double[]> computeMomentInvarVecs() throws InterruptedException, TimeoutException {
        /*System.out.print("Computing moment invariant vectors, stand by");
        ArrayList<double[]> vecs = new ArrayList<>();
        int frameID = 0;
        int dotBoundary = N_FRAME / 5;
        Mat frame = new Mat();
        Mat gray = new Mat();
        boolean successful = cap.read(frame);
        while (successful) {
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
            vecs.add(computeMomentInvarVec(gray));
            if (frameID >= dotBoundary) {
                System.out.print(".");
                dotBoundary += N_FRAME / 5;
            }
            successful = cap.read(frame);
            frameID++;
        }
        System.out.println("OK");
        return vecs;*/

        ArrayList<double[]> vecs = new ArrayList<>();
        Mat frame = new Mat();
        Mat gray = new Mat();
        boolean successful = cap.read(frame);
        while (successful) {
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
            Moments moments = Imgproc.moments(gray);
            Mat hu = new Mat();
            Imgproc.HuMoments(moments, hu);
            double[] vec = {hu.get(0, 0)[0], hu.get(1, 0)[0], hu.get(2, 0)[0]};
            vecs.add(vec);
            successful = cap.read(frame);
        }
        return vecs;
    }

    //计算矩不变向量
    private double[] computeMomentInvarVec(Mat gray) throws InterruptedException, TimeoutException {
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
            throw new TimeoutException("Task \"Compute Moment Invariants\" didn't finish in time."); //太长时间没能计算完毕，抛出异常告知

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
    private double computeMomentInvarDist(double[] momentInvarVec_A, double[] momentInvarVec_B) throws InterruptedException, TimeoutException {
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

//class TimeOutException extends Exception {
//    public TimeOutException(String msg) {
//        super(msg);
//    }
//}