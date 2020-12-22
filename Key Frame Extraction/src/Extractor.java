import java.io.*;
import java.util.*;
import java.util.concurrent.*;

import org.opencv.core.*;
import org.opencv.imgcodecs.*;
import org.opencv.videoio.*;
import org.opencv.imgproc.*;

/**
 * “提取器”主类，提供方法提取指定视频文件的关键帧
 */
public class Extractor implements AutoCloseable {
    private String FILE_PATH;
    private int FPS;
    private VideoCapture cap;
    private int WIDTH;
    private int HEIGHT;
    private double SHOT_BOUNDARY_THRESHOLD = 2E-7; //利用矩不变量法进行镜头边缘检测的阈值，推荐设为2E-7
    private double KEY_FRAME_INVAR_THRESHOLD = 3E-6; //用于通过矩不变量法进行关键帧提取的阈值，推荐设为3E-6
    private double HIST_B_WEIGHT = 1; //计算直方图相似度时，蓝色通道的权重
    private double HIST_G_WEIGHT = 1;
    private double HIST_R_WEIGHT = 1;
    private double KEY_FRAME_KMEANS_THRESHOLD = 0.1; //通过k均值聚类进行关键帧提取的阈值，阈值越大，提取出的关键帧越多。推荐设为0.1
    private String KEY_FRAMES_FOLDER;
    private int N_FRAME;
    private int N_EFFECTIVE_FRAME; //有效帧数：末尾几帧可能读取失败

    private static boolean usingChinese = false;

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    /**
     * 主函数，提供程序入口和交互功能
     */
    public static void main(String[] args) {
        //提示选择语言
        Scanner in = new Scanner(System.in);
        System.out.println("Choose your language:");
        System.out.println("1. English");
        System.out.println("2. 中文");
        System.out.print("> ");
        String choice = in.nextLine();
        if (choice.charAt(0) == '1') usingChinese = false;
        else if (choice.charAt(0) == '2') usingChinese = true;
        else System.out.println("\"" + choice + "\" is not a valid input. Program would use English.");
        //提示输入文件路径
        if (usingChinese) {
            System.out.println("请输入视频文件的路径。");
            System.out.println("注意：路径中请不要包含中文字符，否则可能造成异常。");
        }
        else {
            System.out.println("Input path of the video file.");
            System.out.println("Note: the path should not contain any Chinese characters, otherwise it may cause an error.");
        }
        System.out.print("> ");
        String filePath = in.nextLine();
        File file = new File(filePath);
        if (!file.exists() || !file.isFile()) { //文件不存在，返回
            if (usingChinese)
                System.out.println("警告：\"" + filePath + "\" 不是符合规则的路径，或者文件不存在。");
            else
                System.out.print("Warning: \"" + filePath + "\" is not a valid file path or file does not exist.");
            in.nextLine();
            return;
        }
        //提示输入关键帧输出路径
        if (usingChinese) {
            System.out.println("请输入用于生成关键帧的目录的路径。");
            System.out.println("注意：请输入已经存在的目录的路径，路径应以\"/\"结尾。");
            System.out.println("     你可以按下[Enter]来跳过此步骤，关键帧正将被生成在与程序相同的目录下。");
        }
        else {
            System.out.println("Input the directory path for writing key frames.");
            System.out.println("Note: make sure the folder already exists. Path must end with a \"/\" character.");
            System.out.println("      You can choose to write key frames in the same folder as the program. In such case, just skip by pressing [Enter].");
        }
        System.out.print("> ");
        String folderPath = in.nextLine();
        if (!folderPath.equals("")) {
            file = new File(folderPath);
            if (!file.exists() || !file.isDirectory()) {
                if (usingChinese)
                    System.out.println("警告：\"" + filePath + "\" 不是符合规则的路径，或者目录不存在。");
                else
                    System.out.print("Warning: \"" + folderPath + "\" is not a valid directory path or directory does not exist.");
                in.nextLine();
                return;
            }
        }
        //提示选择关键帧提取方法
        boolean usingKMeans = false;
        if (usingChinese) {
            System.out.println("要使用K-均值聚类法吗？ [Y/N]");
            System.out.println("使用K-均值聚类法，结果将含有更少的非关键帧或重复帧，但耗时可能急剧上升（大约6倍）。");
        }
        else {
            System.out.println("Use K-Means method? [Y/N]");
            System.out.println("By using K-Means, the result would contain less irrelevant or duplicate frames, but " +
                    "time cost would sharply increase (by approximately 6 times).");
        }
        System.out.print("> ");
        choice = in.nextLine();
        if (choice.charAt(0) == 'Y' || choice.charAt(0) == 'y') usingKMeans = true;
        if (usingChinese) {
            if (usingKMeans) System.out.println("使用K-均值聚类");
            else System.out.println("使用矩不变量");
        }
        else {
            if (usingKMeans) System.out.println("Using K-Means.");
            else System.out.println("Using moment invariants.");
        }

        ArrayList<Integer> boundaryFrameIDs = null;
        ArrayList<double[]> momentInvarVecs = null;
        Mat frame = new Mat();
        try (Extractor extractor = new Extractor(filePath, folderPath);) {
            momentInvarVecs = extractor.computeMomentInvarVecs();
            boundaryFrameIDs = extractor.shotBoundaryDetection(momentInvarVecs);
            if (usingKMeans) extractor.keyFrameExtraction_KMeans(boundaryFrameIDs);
            else extractor.keyFrameExtraction_Invar(boundaryFrameIDs, momentInvarVecs);
        }
        catch (Exception ex) {
            if (usingChinese)
                System.out.println("提取关键帧过程中发生异常，异常信息：");
            else
                System.out.println("Error occured while extracting key frames. Error message:");
            System.out.println(ex);
            ex.printStackTrace();
            in.nextLine();
            return;
        }
        if (usingChinese) System.out.print("按下[Enter]退出。");
        else System.out.print("Press [Enter] to exit.");
        in.nextLine();
    }

    /**
     * 构造函数
     *
     * @param filePath        视频文件路径
     * @param keyFramesFolder 输出关键帧的路径，末尾应带有分隔符"/"或"\\"
     * @throws IOException 无法读取视频
     */
    public Extractor(String filePath, String keyFramesFolder) throws IOException {
        FILE_PATH = filePath;
        cap = new VideoCapture(FILE_PATH);
        FPS = (int) cap.get(CVConsts.CAP_PROP_FPS);
        WIDTH = (int) cap.get(CVConsts.CAP_PROP_FRAME_WIDTH);
        HEIGHT = (int) cap.get(CVConsts.CAP_PROP_FRAME_HEIGHT);
        KEY_FRAMES_FOLDER = keyFramesFolder;
        N_EFFECTIVE_FRAME = N_FRAME = (int) cap.get(CVConsts.CAP_PROP_FRAME_COUNT);
        if (N_FRAME < 1) { //读取失败
            throw new IOException("Cannot read video file.");
        }
    }

    /**
     * 计算各帧的矩不变向量。此函数同时也会确定有效帧的数量并更新N_EFFECTIVE_FRAME成员。
     *
     * @return 各帧的矩不变向量
     * @throws InterruptedException 线程被中断
     * @throws TimeoutException     过长时间未能完成计算
     */
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

        int dotBoundary = N_FRAME / 5; //用于模拟进度条
        int nDotsPrinted = 0;

        if (usingChinese) System.out.print("计算矩不变向量");
        else System.out.print("Computing moment invariant vectors");
        //计算矩不变向量
        ArrayList<double[]> vecs = new ArrayList<>();
        Mat frame = new Mat();
        Mat gray = new Mat();
        boolean successful = cap.read(frame);
        int frameID = 0;
        while (successful) {
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY); //利用灰度图计算
            Moments moments = Imgproc.moments(gray);
            Mat hu = new Mat();
            Imgproc.HuMoments(moments, hu);
            double[] vec = {hu.get(0, 0)[0], hu.get(1, 0)[0], hu.get(2, 0)[0]};
            vecs.add(vec);

            //更新进度条
            if (frameID > dotBoundary) {
                System.out.print(".");
                dotBoundary += N_FRAME / 5;
                nDotsPrinted++;
            }

            successful = cap.read(frame);
            frameID++;
        }

        N_EFFECTIVE_FRAME = vecs.size(); //末尾几帧可能读取失败，更新有效帧数

        while (nDotsPrinted < 5) {
            System.out.print(".");
            nDotsPrinted++;
        }
        System.out.println("OK");

        return vecs;
    }

    /**
     * 镜头边缘检测
     *
     * @param vecs 各帧的矩不变向量
     * @return 镜头边缘帧的帧号组成的列表
     * @throws InterruptedException 线程被中断
     * @throws TimeoutException     过长时间未能完成计算
     */
    public ArrayList<Integer> shotBoundaryDetection(ArrayList<double[]> vecs) throws InterruptedException, TimeoutException {
        if (usingChinese) System.out.print("检测镜头边界");
        else System.out.print("Detecting shot boundaries..");

        ArrayList<Integer> boundaryFrameIDs = new ArrayList<>();
        boundaryFrameIDs.add(0); //第一帧直接作为边界
        int lastBoundaryID = 0;
        double[] vecOfLast = null;
        double[] vecOfThis = null;
        for (int i = 1; i <= vecs.size() - 1; i++) {
            vecOfLast = vecs.get(i - 1);
            vecOfThis = vecs.get(i);
            double momentInvarDist = computeMomentInvarDist(vecOfLast, vecOfThis);
            //当前帧与前一帧矩不变向量的距离大于阈值，且相隔超过半秒时，加入边界列表
            if (momentInvarDist > SHOT_BOUNDARY_THRESHOLD && i - lastBoundaryID > FPS / 2) {
                boundaryFrameIDs.add(i);
                lastBoundaryID = i;
            }
        }

        System.out.println("...OK");

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

    /**
     * 提取关键帧，基于矩不变量法
     *
     * @param boundaryIDs 边界帧帧号组成的列表
     * @param vecs        各帧矩不变向量组成的列表
     * @throws InterruptedException 线程被中断
     * @throws TimeoutException     过长时间未能完成计算
     */
    public void keyFrameExtraction_Invar(ArrayList<Integer> boundaryIDs, ArrayList<double[]> vecs) throws InterruptedException, TimeoutException {
        int dotBoundary = N_EFFECTIVE_FRAME / 5;
        int nDotsPrinted = 0;

        if (usingChinese) System.out.print("提取关键帧");
        else System.out.print("Extracting key frames");
        //各个镜头独立处理
        for (int i = 0; i <= boundaryIDs.size() - 2; i++) {
            int left = boundaryIDs.get(i);
            int right = boundaryIDs.get(i + 1);
            inShotKeyFrameExtraction_Invar(vecs, left, right);

            if (right >= dotBoundary) {
                System.out.print(".");
                dotBoundary += N_EFFECTIVE_FRAME / 5;
                nDotsPrinted++;
            }
        }

        while (nDotsPrinted < 5) {
            System.out.print(".");
            nDotsPrinted++;
        }

        //单独处理最后一个镜头
        inShotKeyFrameExtraction_Invar(vecs, boundaryIDs.get(boundaryIDs.size() - 1), N_EFFECTIVE_FRAME);
        System.out.println("OK");
    }

    /**
     * 镜头内关键帧提取，基于矩不变量法。直接在输出目录写入关键帧，无返回值
     *
     * @param vecs  各帧矩不变向量组成的列表
     * @param left  镜头左边界的帧号（含）
     * @param right 镜头右边界的帧号（不含）
     * @throws InterruptedException 线程被中断
     * @throws TimeoutException     过长时间未能完成计算
     */
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
        //如果第一帧不是暗帧则输出
        if (!isDarkFrame(frame)) Imgcodecs.imwrite(KEY_FRAMES_FOLDER + lastKeyFrameID + ".jpg", frame);
        double[] vecOfLastKeyFrame = vecs.get(lastKeyFrameID);
        //遍历镜头中的各帧
        for (int i = lastKeyFrameID + 1; i <= right - 1; i++) {
            double[] vecOfThis = vecs.get(i);
            double dist = computeMomentInvarDist(vecOfLastKeyFrame, vecOfThis);
            //与上一关键帧距离超过阈值，且相隔半秒以上时作为关键帧输出
            if (dist > KEY_FRAME_INVAR_THRESHOLD && i - lastKeyFrameID > FPS / 2) {
                cap.set(CVConsts.CAP_PROP_POS_FRAMES, i);
                successful = cap.read(frame);
                if (!successful) continue; //读取失败，舍弃此帧
                Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
                if (!isDarkFrame(gray)) {
                    Imgcodecs.imwrite(KEY_FRAMES_FOLDER + i + ".jpg", frame);
                    lastKeyFrameID = i;
                    vecOfLastKeyFrame = vecOfThis;
                }
            }
        }
    }

    /**
     * 关键帧提取，基于K均值聚类
     *
     * @param boundaryIDs 边界帧帧号组成的列表
     * @throws InterruptedException 线程被中断
     * @throws TimeoutException     过长时间未能完成计算
     */
    private void keyFrameExtraction_KMeans(ArrayList<Integer> boundaryIDs) throws InterruptedException, TimeoutException {
        /*System.out.print("Extracting key frames");
        int nDotsPrinted = 0;
        int dotBoundary = N_EFFECTIVE_FRAME / 5;
        for (int i = 0; i <= boundaryIDs.size() - 2; i++) {
            int left = boundaryIDs.get(i);
            int right = boundaryIDs.get(i + 1);
            inShotKeyFrameExtraction_KMeans(left, right);
            if (right >= dotBoundary) {
                System.out.print(".");
                dotBoundary += N_EFFECTIVE_FRAME / 5;
                nDotsPrinted++;
            }
        }
        if (nDotsPrinted < 5) System.out.print(".");
        inShotKeyFrameExtraction_KMeans(boundaryIDs.get(boundaryIDs.size() - 1), N_EFFECTIVE_FRAME);
        System.out.println("OK");*/

        /*long startTime = System.currentTimeMillis();*/

        if (usingChinese) System.out.print("提取关键帧");
        else System.out.print("Extracting key frames");

        //分镜头并行计算颜色直方图
        int[][][] hists = new int[N_EFFECTIVE_FRAME][3][256]; //[帧号][通道][像素值]
        ExecutorService executor = Executors.newCachedThreadPool();
        for (int i = 0; i <= boundaryIDs.size() - 2; i++) {
            int left = boundaryIDs.get(i);
            int right = boundaryIDs.get(i + 1);
            executor.execute(new HistComputeTask(hists, left, right, false));
        }
        executor.execute(new HistComputeTask(hists, boundaryIDs.get(boundaryIDs.size() - 1), N_EFFECTIVE_FRAME, true));
        executor.shutdown();
        boolean successful = executor.awaitTermination(2 * N_FRAME / FPS, TimeUnit.SECONDS);
        if (!successful) throw new TimeoutException("Task \"Compute Histograms\" didn't finish in time.");

        /*long endTime = System.currentTimeMillis();
        System.out.println("\tHistogram: " + (endTime - startTime) + "ms.");*/

        //分镜头提取关键帧
        for (int i = 0; i <= boundaryIDs.size() - 2; i++) {
            int left = boundaryIDs.get(i);
            int right = boundaryIDs.get(i + 1);
            inShotKeyFrameExtraction_KMeans(hists, left, right);
        }
        inShotKeyFrameExtraction_KMeans(hists, boundaryIDs.get(boundaryIDs.size() - 1), N_EFFECTIVE_FRAME);

        System.out.println("OK");
    }

    /**
     * 镜头内关键帧提取，基于K均值聚类
     *
     * @param hists 各帧的颜色直方图
     * @param left  镜头左边界的帧号（含）
     * @param right 镜头右边界的帧号（不含）
     */
    private void inShotKeyFrameExtraction_KMeans(int[][][] hists, int left, int right) {
        /*cap.set(CVConsts.CAP_PROP_POS_FRAMES, left);
        Mat frame = new Mat();
        cap.read(frame);
        ArrayList<Mat> frames = new ArrayList<>();
        frames.add(frame);
        MatOfInt channels = new MatOfInt(0, 1, 2);
        Mat mask = new Mat();
        Mat hists = new Mat();
        MatOfInt histSize = new MatOfInt(256, 256, 256);
        MatOfFloat ranges = new MatOfFloat(3, 2, CvType.CV_32F);
        float[] rangeLeft = {0};
        float[] rangeRight = {256};
        ranges.put(0, 0, rangeLeft);
        ranges.put(0, 1, rangeRight);
        ranges.put(1, 0, rangeLeft);
        ranges.put(1, 1, rangeRight);
        ranges.put(2, 0, rangeLeft);
        ranges.put(2, 1, rangeRight);
        Imgproc.calcHist(frames, channels, mask, hists, histSize, ranges);
        System.out.println(hists.dump());*/

        /*long startTime = System.currentTimeMillis();

        cap.set(CVConsts.CAP_PROP_POS_FRAMES, left);
        Mat frame = new Mat();
        int nFrames = right - left;
        //计算颜色直方图
        int[][][] hists = new int[nFrames][3][256]; //[帧][通道][像素值]
        for (int i = 0; i <= nFrames - 1; i++) {
            cap.read(frame);

            for (int y = 0; y <= HEIGHT - 1; y++) {
                for (int x = 0; x <= WIDTH - 1; x++) {
                    int B = (int) frame.get(y, x)[0];
                    int G = (int) frame.get(y, x)[1];
                    int R = (int) frame.get(y, x)[2];
                    hists[i][0][B]++;
                    hists[i][1][G]++;
                    hists[i][2][R]++;
                }
            }

            HistComputeAction masterAction = new HistComputeAction(frame, hists[i], 0, WIDTH, 0, HEIGHT);
            ForkJoinPool pool = new ForkJoinPool();
            pool.invoke(masterAction);
        }

        long endTime = System.currentTimeMillis();
        System.out.println("\tColor Histogram: " + (endTime - startTime) + "ms.");
        startTime = endTime;*/

        /*long startTime, endTime;
        startTime = System.currentTimeMillis();*/

        Mat frame = new Mat();
        int nFrames = right - left;
        int frameID;
        //聚类分析
        ArrayList<HistClusterCenter> centers = new ArrayList<>();
        for (int i = 0; i <= nFrames - 1; i++) {
            frameID = i + left;
            int[][] hist = hists[frameID];
            HistClusterCenter nearest = findNearestCenter(centers, hist);
            if (nearest == null) { //建立新中心
                HistClusterCenter newCenter = new HistClusterCenter(new Histogram(frameID, hist));
                centers.add(newCenter);
            }
            else { //加入最近中心，调整中心位置
                nearest.members.add(new Histogram(frameID, hist));
                nearest.updateHist();
            }
        }

        /*endTime = System.currentTimeMillis();
        System.out.println("\tClustering: " + (endTime - startTime) + "ms.");
        startTime = endTime;*/

        //进一步筛选关键帧并输出
        TreeSet<Integer> keyFrameIds = new TreeSet<>();
        //将各个聚类最靠近中心的帧加入关键帧列表
        for (HistClusterCenter center : centers) {
            frameID = center.findNearest();
            keyFrameIds.add(frameID);
            /*cap.set(CVConsts.CAP_PROP_POS_FRAMES, frameID);
            cap.read(frame);
            Mat gray = new Mat();
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
            if (!isDarkFrame(gray)) {
                Imgcodecs.imwrite(KEY_FRAMES_FOLDER + frameID + ".jpg", frame);
            }*/
        }
        boolean isDark = true;
        int firstKeyFrameID = 0;
        Mat gray = new Mat();
        //提取第一个非暗关键帧
        while (!keyFrameIds.isEmpty()) {
            firstKeyFrameID = keyFrameIds.first();
            cap.set(CVConsts.CAP_PROP_POS_FRAMES, firstKeyFrameID);
            cap.read(frame);
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
            isDark = isDarkFrame(gray);
            if (!isDark) {
                Imgcodecs.imwrite(KEY_FRAMES_FOLDER + firstKeyFrameID + ".jpg", frame);
                break;
            }
            else keyFrameIds.remove(firstKeyFrameID);
        }
        int lastKeyFrameID = firstKeyFrameID;
        //对列表中各帧依据亮度和位置进行筛选
        for (int id : keyFrameIds) {
            if (id - lastKeyFrameID < FPS / 2) continue;
            cap.set(CVConsts.CAP_PROP_POS_FRAMES, id);
            cap.read(frame);
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
            if (!isDarkFrame(gray)) {
                Imgcodecs.imwrite(KEY_FRAMES_FOLDER + id + ".jpg", frame);
                lastKeyFrameID = id;
            }
        }

        /*endTime = System.currentTimeMillis();
        System.out.println("\tFiltering: " + (endTime - startTime) + "ms.\n");*/
    }

    //todo 自适应暗帧检测

    /**
     * 判断一个帧是否过暗
     *
     * @param gray 灰度图
     * @return 帧过暗时返回true
     */
    private boolean isDarkFrame(Mat gray) {
        long sum = 0;
        for (int y = 0; y <= HEIGHT - 1; y++) {
            for (int x = 0; x <= WIDTH - 1; x++) {
                sum += gray.get(y, x)[0];
            }
        }
        return sum <= 255 / 8 * WIDTH * HEIGHT;
    }

    /**
     * 计算两幅灰度图矩不变向量的欧拉距离
     *
     * @param momentInvarVec_A 灰度图A的矩不变向量
     * @param momentInvarVec_B 灰度图B的矩不变向量
     * @return 矩不变向量的欧拉距离
     */
    private double computeMomentInvarDist(double[] momentInvarVec_A, double[] momentInvarVec_B) {
        double deltaX = momentInvarVec_A[0] - momentInvarVec_B[0];
        double deltaY = momentInvarVec_A[1] - momentInvarVec_B[1];
        double deltaZ = momentInvarVec_A[2] - momentInvarVec_B[2];
        return Math.pow(deltaX, 2) + Math.pow(deltaY, 2) + Math.pow(deltaZ, 2);
    }

    /**
     * 计算图像颜色直方图的相似度
     *
     * @param histA 图像A的颜色直方图
     * @param histB 图像B的颜色直方图
     * @return 颜色直方图的相似度
     */
    private double computeHistSim(int[][] histA, int[][] histB) {
        double[] sims = new double[3]; //三个通道的相似度
        int SUM = WIDTH * HEIGHT; //像素数
        for (int i = 0; i <= 255; i++) {
            sims[0] += Math.min(histA[0][i], histB[0][i]);
            sims[1] += Math.min(histA[1][i], histB[1][i]);
            sims[2] += Math.min(histA[2][i], histB[2][i]);
        }
        sims[0] /= SUM;
        sims[1] /= SUM;
        sims[2] /= SUM;
        return (sims[0] * HIST_B_WEIGHT + sims[1] * HIST_G_WEIGHT + sims[2] * HIST_R_WEIGHT) / (HIST_B_WEIGHT + HIST_G_WEIGHT + HIST_R_WEIGHT);
    }

    /**
     * 寻找与给定直方图最近的聚类中心，找不到相似度超过阈值的聚类中心时返回null
     *
     * @param centers 聚类中心组成的列表
     * @param hist    颜色直方图
     * @return 最近的聚类中心的引用或空引用
     */
    private HistClusterCenter findNearestCenter(ArrayList<HistClusterCenter> centers, int[][] hist) {
        if (centers.size() < 1) return null;
        HistClusterCenter nearest = null;
        double maxSim = -1;
        for (int i = 0; i <= centers.size() - 1; i++) {
            HistClusterCenter center = centers.get(i);
            double sim = computeHistSim(center.hist, hist);
            if (sim > maxSim && sim > KEY_FRAME_KMEANS_THRESHOLD) {
                maxSim = sim;
                nearest = centers.get(i);
            }
        }
        return nearest;
    }

    /**
     * 计算灰度图的重心坐标
     *
     * @param gray 灰度图
     * @param sum  所有像素值之和
     * @return 重心坐标
     */
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

    /**
     * 释放资源
     */
    public void close() {
        cap.release();
    }

    /**
     * 并行计算一幅灰度图的矩不变向量（因性能不佳被搁置）
     *
     * @param gray 灰度图
     * @return 灰度图的矩不变向量
     * @throws InterruptedException 线程被中断
     * @throws TimeoutException     过长时间未能完成计算
     */
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

        //计算矩不变向量
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

    /**
     * 计算一幅灰度图的矩不变量（因性能不佳被搁置）
     *
     * @param gray 灰度图
     * @param p    x的次数
     * @param q    y的次数
     * @param xBar 重心的横坐标
     * @param yBar 重心的纵坐标
     * @param sum  所有像素值之和
     * @return 矩不变量
     */
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

    /**
     * 计算灰度图的图像矩（因性能不佳被搁置）
     *
     * @param gray 灰度图
     * @param p    x的次数
     * @param q    y的次数
     * @return 图像矩
     */
    private double computeMoment(Mat gray, double p, double q) {
        double m = 0;
        for (int y = 0; y <= HEIGHT - 1; y++) { //Mat的行数等于图像的高度
            for (int x = 0; x <= WIDTH - 1; x++) {
                m += Math.pow(x, p) * Math.pow(y, q) * gray.get(y, x)[0];
            }
        }
        return m;
    }

    /**
     * 用于并行计算颜色直方图的内部类
     */
    private class HistComputeTask implements Runnable {
        private int[][][] hists; //用于存放结果的直方图数组
        private int start; //任务的左边界（含），用于指示任务范围
        private int end; //任务的右边界（不含）
        private boolean printDots; //是否要模拟进度条

        private HistComputeTask(int[][][] hists, int start, int end, boolean printDots) {
            this.hists = hists;
            this.start = start;
            this.end = end;
            this.printDots = printDots;
        }

        public void run() {
            int nFrames = end - start;
            int dotBoundary = nFrames / 5;
            int nDotsPrinted = 0;

            Mat frame = new Mat();
            for (int i = start; i <= end - 1; i++) {
                synchronized (cap) {
                    cap.set(CVConsts.CAP_PROP_POS_FRAMES, i);
                    cap.read(frame);
                }

                for (int y = 0; y <= HEIGHT - 1; y++) {
                    for (int x = 0; x <= WIDTH - 1; x++) {
                        int B = (int) frame.get(y, x)[0];
                        int G = (int) frame.get(y, x)[1];
                        int R = (int) frame.get(y, x)[2];
                        hists[i][0][B]++;
                        hists[i][1][G]++;
                        hists[i][2][R]++;
                    }
                }

                if (printDots && i - start >= dotBoundary) {
                    System.out.print(".");
                    dotBoundary += nFrames / 5;
                    nDotsPrinted++;
                }

            }

            if (printDots) {
                while (nDotsPrinted < 5) {
                    System.out.print(".");
                    nDotsPrinted++;
                }
            }
        }
    }

    /**
     * “直方图聚类中心”内部类
     */
    public class HistClusterCenter {
        private int[][] hist; //中心的直方图
        private ArrayList<Histogram> members;

        public HistClusterCenter(Histogram histObj) {
            this.hist = new int[histObj.hist.length][histObj.hist[0].length];
            for (int i = 0; i <= this.hist.length - 1; i++) {
                System.arraycopy(histObj.hist[i], 0, this.hist[i], 0, this.hist[0].length);
            }
            members = new ArrayList<>();
            members.add(histObj);
        }

        //更新中心的直方图，应在加入新成员后调用
        public void updateHist() {
            int nMembers = members.size();
            for (int channel = 0; channel <= 2; channel++) {
                for (int v = 0; v <= 255; v++) {
                    hist[channel][v] = (int) ((nMembers - 1.0) / nMembers * hist[channel][v] + 1.0 / nMembers * members.get(nMembers - 1).hist[channel][v]);
                }
            }
        }

        //寻找距离中心最近的帧，返回帧ID
        public int findNearest() {
            Histogram nearest = members.get(0);
            double maxSim = computeHistSim(hist, nearest.hist);
            for (int i = 1; i <= members.size() - 1; i++) {
                Histogram member = members.get(i);
                double sim = computeHistSim(hist, member.hist);
                if (sim > maxSim) {
                    maxSim = sim;
                    nearest = member;
                }
            }
            return nearest.frameID;
        }
    }

    /**
     * “直方图”内部类，以帧ID和直方图作为成员。其实例用作HistClusterCenter.members的元素，方便找出最接近中心的帧ID。
     */
    public class Histogram {
        private int frameID;
        private int[][] hist;

        public Histogram(int frameID, int[][] hist) {
            this.frameID = frameID;
            this.hist = hist;
        }
    }

    /**
     * 用于并行计算矩不变量的内部类（因性能不佳被搁置）
     */
    private class MomentInvarComputeTask implements Runnable {
        private Mat gray;
        private double xBar;
        private double yBar;
        private double sum; //灰度图像素值之和
        private double p;
        private double q;
        private double[] result; //存放结果的数组
        private int index; //在数组的哪个位置写入结果

        private MomentInvarComputeTask(Mat gray, double p, double q, double xBar, double yBar, double sum, double[] result, int index) {
            this.gray = gray;
            this.xBar = xBar;
            this.yBar = yBar;
            this.sum = sum;
            this.p = p;
            this.q = q;
            this.result = result;
            this.index = index;
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

    /**
     * 用于并行计算颜色直方图的内部类（由于性能不佳被搁置）
     */
    private class HistComputeAction extends RecursiveAction {
        private Mat frame;
        private int[][] hist; //用于存放计算结果的数组
        private int yStart; //纵坐标的左边界，用于指示任务范围
        private int yEnd;
        private int xStart;
        private int xEnd;

        private HistComputeAction(Mat frame, int[][] hist, int xStart, int xEnd, int yStart, int yEnd) {
            this.frame = frame;
            this.hist = hist;
            this.xStart = xStart;
            this.xEnd = xEnd;
            this.yStart = yStart;
            this.yEnd = yEnd;
        }

        public void compute() {
            //任务规模较小，串行解决
            if (xEnd - xStart <= 10 || yEnd - yStart <= 10) {
                for (int x = xStart; x <= xEnd - 1; x++) {
                    for (int y = yStart; y <= yEnd - 1; y++) {
                        int B = (int) frame.get(y, x)[0];
                        int G = (int) frame.get(y, x)[1];
                        int R = (int) frame.get(y, x)[2];
                        synchronized (hist) {
                            hist[0][B]++;
                            hist[1][G]++;
                            hist[2][R]++;
                        }
                    }
                }
            }
            //任务规模较大，拆分成四个子任务
            else {
                int xMid = (xStart + xEnd) / 2;
                int yMid = (yStart + yEnd) / 2;
                HistComputeAction leftUp = new HistComputeAction(frame, hist, xStart, xMid, yStart, yMid);
                HistComputeAction rightUp = new HistComputeAction(frame, hist, xMid, xEnd, yStart, yMid);
                HistComputeAction leftDown = new HistComputeAction(frame, hist, xStart, xMid, yMid, yEnd);
                HistComputeAction rightDown = new HistComputeAction(frame, hist, xMid, xEnd, yMid, yEnd);
                invokeAll(leftUp, leftDown, rightUp, rightDown);
            }
        }
    }
}

//class TimeOutException extends Exception {
//    public TimeOutException(String msg) {
//        super(msg);
//    }
//}