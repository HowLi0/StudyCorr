#include "StudyCorr.h"
#include <fstream>

int StudyCorr::CalibrationIndex = 0;
int StudyCorr::ComputeIndex = 0;

StudyCorr::StudyCorr(QWidget* parent)
	: QMainWindow(parent),
	chessToolBar(nullptr),
	computeToolBar(nullptr),
	CalibrationButton(nullptr),
	ComputeButton(nullptr),
	StartCalibrationButton(nullptr),
	StartComputeButton(nullptr),
	stepSizeSpinBox(nullptr),
	subSizeSpinBox(nullptr),
	squareSizeSpinBox(nullptr),
	rowsSpinBox(nullptr),
	colsSpinBox(nullptr),
	item1(nullptr),
	item2(nullptr),
	img1TextItem(nullptr),
	img2TextItem(nullptr),
	view1(nullptr),
	view2(nullptr),
	drawable1(nullptr),
	TabWidget(nullptr),
	TreeWidget1(nullptr),
	TreeWidget2(nullptr),
	calibrationDialog(nullptr),
	computeDialog(nullptr),
	chessCalibration(nullptr),
	timer(nullptr),
	hasRunCalibrationToolbars(false),
	hasRunComputeToolbars(false),
	currentFrameIndex(0)
{
	ui.setupUi(this);
	resize(1200, 800);
	this->CalibrationIndex = 0;
	this->ComputeIndex = 0;
	SetupUi(this->CalibrationIndex, this->ComputeIndex);
}

StudyCorr::~StudyCorr()
{
	delete calibrationFactory;
}

void StudyCorr::SetupUi(int CalibrationIndex, int ComputeIndex)
{
	//****************************************************菜单栏****************************************************//
	//菜单栏创建,菜单栏最多只能有一个
	QMenuBar* Bar = menuBar();//菜单栏默认放在树中
	setMenuBar(Bar);//将菜单栏放在窗口中
	//创建菜单
	QMenu* FileMenu = Bar->addMenu("文件");
	QMenu* CalibrationMenu = Bar->addMenu("标定");
	QMenu* ComputeMenu = Bar->addMenu("计算");
	QMenu* DataMenu = Bar->addMenu("数据");
	QMenu* HelpMenu = Bar->addMenu("帮助");
	//创建菜单项
	QAction* NewAction = FileMenu->addAction("新建");
	NewAction->setIcon(QIcon(":/icon/1.png"));
	FileMenu->addSeparator();//添加分割线
	QAction* OpenAction = FileMenu->addAction("打开");
	OpenAction->setIcon(QIcon(":/icon/6.png"));

	// 连接信号和槽
	connect(NewAction, &QAction::triggered, this, &StudyCorr::CreateNewProject);
	connect(OpenAction, &QAction::triggered, this, &StudyCorr::OpenExistingProject);
	connect(DataMenu, &QMenu::aboutToShow, this, &StudyCorr::downloadData);

	//****************************************************工具栏****************************************************//
	//添加工具栏
	QToolBar* ToolBar = new QToolBar(this);//工具栏默认不放在树中
	//addToolBar(ToolBar)
	addToolBar(Qt::TopToolBarArea, ToolBar);//更改停靠区域
	//设置允许停靠区域
	ToolBar->setAllowedAreas(Qt::TopToolBarArea | Qt::LeftToolBarArea | Qt::RightToolBarArea);
	//ToolBar->setMovable(false);
	//ToolBar->setFixedHeight(50);

	//工具栏加工具项
	ToolBar->addAction(NewAction);//接受菜单项，新建文件夹
	ToolBar->addSeparator();
	ToolBar->addAction(OpenAction);//接受菜单项，打开文件夹
	ToolBar->addSeparator();
	CalibrationButton = ToolBar->addAction("标定");//标定
	CalibrationButton->setIcon(QIcon(":/icon/7.png"));
	ToolBar->addSeparator();
	ComputeButton = ToolBar->addAction("计算");//计算
	ComputeButton->setIcon(QIcon(":/icon/8.png"));

	//连接信号和槽
	connect(CalibrationButton, &QAction::triggered, [=]() {
		this->CalibrationButtonClicked(this->CalibrationIndex);
		// 只有在第一次点击时才运行
		if (!hasRunCalibrationToolbars) {
			CalibrationToolBar();
			ChessToolBar();
			hasRunCalibrationToolbars = true;  // 设置标志为true，防止再次调用
		}
		this->CalibrationIndex++;
		});

	connect(ComputeButton, &QAction::triggered, [=]() {
		this->ComputeButtonClicked(this->ComputeIndex);
		if (!hasRunComputeToolbars)
		{
			ComputeToolBar();
			hasRunComputeToolbars = true;
		}
		this->ComputeIndex++;
		});

	//	//****************************************************状态栏****************************************************//
		//状态栏,最多有一个
	QStatusBar* stbar = new QStatusBar(this);
	//设置到窗口中
	setStatusBar(stbar);
	//放标签
	QLabel* label = new QLabel("NUAA", this);
	stbar->addWidget(label);
	QLabel* label2 = new QLabel("有问题请联系：476459099@qq.com", this);
	stbar->addPermanentWidget(label2);

	//****************************************************浮动窗口****************************************************//
	//铆接部件（浮动窗口），可以有多个
	QDockWidget* dock = new QDockWidget("工作区", this);
	dock->setFixedWidth(100);
	this->TabWidget = new QTabWidget(dock);
	// Create the first tree widget and add it to the first tab
	this->TreeWidget1 = new QTreeWidget(TabWidget);
	TreeWidget1->setHeaderHidden(true); // 隐藏顶部的header
	TabWidget->addTab(TreeWidget1, tr("标定"));

	// Create the second tree widget and add it to the second tab
	this->TreeWidget2 = new QTreeWidget(TabWidget);
	TreeWidget2->setHeaderHidden(true); // 隐藏顶部的header
	TabWidget->addTab(TreeWidget2, tr("计算"));

	dock->setWidget(TabWidget);
	addDockWidget(Qt::LeftDockWidgetArea, dock);

	//****************************************************图像显示/中心控件****************************************************//

// 创建一个 QGraphicsScene 以包含图像
	this->scene1 = new QGraphicsScene(this);
	this->scene2 = new QGraphicsScene(this);

	// 加载图像并添加到场景中
	this->item1 = new CustomPixmapItem();
	this->item2 = new CustomPixmapItem();
	item1->setAcceptHoverEvents(true);
	item2->setAcceptHoverEvents(true);
	scene1->addItem(item1);
	scene2->addItem(item2);
	scene1->setSceneRect(item1->boundingRect());
	scene2->setSceneRect(item2->boundingRect());
	drawable1 = new Drawable(item1);
	static_cast<CustomPixmapItem*>(item1)->setDrawable(drawable1);


	// 创建一个 QGraphicsView 以显示场景
	this->view1 = new QGraphicsView(scene1, this);
	this->view2 = new QGraphicsView(scene2, this);
	view1->fitInView(scene1->sceneRect(), Qt::KeepAspectRatio);
	view2->fitInView(scene2->sceneRect(), Qt::KeepAspectRatio);
	view1->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
	view1->setResizeAnchor(QGraphicsView::AnchorViewCenter);
	view2->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
	view2->setResizeAnchor(QGraphicsView::AnchorViewCenter);

	//显示图像名称
	this->img1TextItem = new QGraphicsTextItem();
	scene1->addItem(img1TextItem);
	this->img2TextItem = new QGraphicsTextItem();
	scene2->addItem(img2TextItem);

	// 创建一个水平布局并添加两个 QGraphicsView 和分割线
	// 添加分割线，这里选择一个竖直方向的分割线
	QFrame* line = new QFrame(this);
	line->setFrameShape(QFrame::VLine); // VLine是垂直线，HLine是水平线
	line->setFrameShadow(QFrame::Sunken); // 设置阴影效果
	QHBoxLayout* layout = new QHBoxLayout();
	layout->addWidget(view1);
	layout->addWidget(line);
	layout->addWidget(view2);

	// 创建一个中央部件并设置布局
	QWidget* centralWidget = new QWidget(this);
	centralWidget->setLayout(layout);

	// 使视图适应图像大小
	//view->fitInView(item, Qt::KeepAspectRatio);

	// 设置 QGraphicsView 为中心窗口部件
	setCentralWidget(centralWidget);
}

//创建的新的工程文件
void StudyCorr::CreateNewProject()
{
	// 使用 QFileDialog 选择文件夹路径
	QString folder = QFileDialog::getExistingDirectory(this, "选择新建的项目文件夹");

	if (!folder.isEmpty()) {
		QDir dir(folder);

		// 生成新的工程文件夹路径
		QString projectFolder = folder + "/NewProject"; // 可以修改为用户输入的名称
		if (dir.mkpath(projectFolder)) {
			// 在新建文件夹中创建一个 .compj 文件
			QString compjFile = projectFolder + "/project.compj";
			QFile file(compjFile);

			if (file.open(QIODevice::WriteOnly)) {
				// 创建一个基本的 JSON 对象保存项目文件信息
				QJsonObject projectJson;
				projectJson["projectName"] = "NewProject";
				projectJson["projectPath"] = projectFolder;
				projectJson["createdDate"] = QDateTime::currentDateTime().toString();

				// 写入 JSON 数据到 .compj 文件
				QJsonDocument doc(projectJson);
				file.write(doc.toJson());
				file.close();

				QMessageBox::information(this, "成功", "新建项目成功！");
			}
			else {
				QMessageBox::critical(this, "失败", "创建项目文件失败！");
			}
		}
		else {
			QMessageBox::critical(this, "失败", "创建新项目文件夹失败！");
		}
	}
}

//打开工程文件
void StudyCorr::OpenExistingProject()
{
	// 使用 QFileDialog 选择一个 .compj 文件
	QString filePath = QFileDialog::getOpenFileName(this, "选择项目文件", "", "项目文件 (*.compj)");

	if (!filePath.isEmpty()) {
		QFile file(filePath);
		if (file.open(QIODevice::ReadOnly)) {
			// 读取文件内容并解析为 JSON
			QByteArray data = file.readAll();
			QJsonDocument doc = QJsonDocument::fromJson(data);
			QJsonObject projectJson = doc.object();

			QString projectName = projectJson["projectName"].toString();
			QString projectPath = projectJson["projectPath"].toString();
			QString createdDate = projectJson["createdDate"].toString();

			// 可以根据读取的项目数据进行其他操作
			QMessageBox::information(this, "项目已打开",
				"项目名称: " + projectName + "\n路径: " + projectPath + "\n创建日期: " + createdDate);
		}
		else {
			QMessageBox::critical(this, "错误", "无法打开项目文件！");
		}
	}
}

//****************************************************标定****************************************************//
//点击标定按钮
void StudyCorr::CalibrationButtonClicked(int CalibrationIndex)
{
	qDebug() << "Button clicked, opening dialog";

	CalibrationDialog* dialog = new CalibrationDialog(this, CalibrationIndex);

	// 点击加载文件
	connect(dialog->TableWidget->horizontalHeader(), &QHeaderView::sectionClicked, dialog, &CalibrationDialog::OnHeaderClicked);

	// Ensure dialog gets deleted when closed
	dialog->setAttribute(Qt::WA_DeleteOnClose);

	// Connect accepted signal to onOkClicked slot
	connect(dialog, &CalibrationDialog::accepted, this, &StudyCorr::CalibrationOKButtonClicked);

	//要获取dialog的信息，需要在主窗口设置一个变量储存，要不然dialog关闭信息就消失了
	//非常重要，坑了我两天。。。。。。。
	calibrationDialog = dialog;

	// Show the dialog
	dialog->show();

	// 手动更新主界面->你到底更不更新啊，还闪退，玩不起别玩
	this->update();
}

//点击CalibrationDialog中的确定按钮
void StudyCorr::CalibrationOKButtonClicked()
{
	qDebug() << "Dialog accepted, running onOkClicked";
	if (calibrationDialog) {
		LeftCameraFilePath.clear();
		RightCameraFilePath.clear();
		LeftCameraFilePath = calibrationDialog->GetLeftFilePath();
		RightCameraFilePath = calibrationDialog->GetRightFilePath();// 获取文件名列表
		//qDebug() << "LeftCameraFilePath: " << LeftCameraFilePath;
		calibrationImageFiles[CalibrationIndex] = qMakePair(LeftCameraFilePath, RightCameraFilePath);
		// 添加顶层项
		QTreeWidgetItem* CalibrationItem = new QTreeWidgetItem(QStringList(calibrationDialog->NameLineEdit->text()));
		TreeWidget1->addTopLevelItem(CalibrationItem);
		CalibrationItem->setTextAlignment(0, Qt::AlignLeft);// 设置列文本居中

		for (int row = 0; row < calibrationDialog->TableWidget->rowCount(); ++row) {
			QTableWidgetItem* item = calibrationDialog->TableWidget->item(row, 1);
			if (item) {
				// 获取文件名
				QString FileName = item->text();

				// 添加子节点
				QTreeWidgetItem* CalibrationImage = new QTreeWidgetItem(QStringList(FileName));
				CalibrationItem->addChild(CalibrationImage);

				// 设置子项文本居中
				CalibrationImage->setTextAlignment(0, Qt::AlignLeft);
			}
		}
		TreeWidget1->update();  // 强制刷新 QTreeWidget
	}
}

//创建标定工具栏
void StudyCorr::CalibrationToolBar()
{
	//添加标定工具栏
	QToolBar* CalibrationToolBar = new QToolBar(this);//工具栏默认不放在树中
	//addToolBar(ToolBar)
	addToolBar(Qt::TopToolBarArea, CalibrationToolBar);//更改停靠区域
	//设置允许停靠区域
	CalibrationToolBar->setAllowedAreas(Qt::TopToolBarArea);
	//CalibrationToolBar->setMovable(false);

	//工具栏加工具项
	QComboBox* ComboBox = new QComboBox(this);//添加下拉框
	// 向下拉框中添加选项
	ComboBox->addItem("chess");
	ComboBox->addItem("circle");
	ComboBox->addItem("code");
	// 将下拉框添加到工具栏
	CalibrationToolBar->addWidget(ComboBox);
	CalibrationToolBar->addSeparator();

	//连接信号和槽
	// 点击计算按钮隐藏标定工具栏
	connect(ComputeButton, &QAction::triggered, [=]() {
		CalibrationToolBar->hide(); });
	// 连接下拉框选择变化的信号
	connect(ComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), [=](int index) {
		switch (index) {
		case 0:
			qDebug() << "Executing command for Option 1";
			ChessToolBar();
			break;
		case 1:
			qDebug() << "Executing command for Option 2";
			chessToolBar->hide();
			break;
		case 2:
			qDebug() << "Executing command for Option 3";
			chessToolBar->hide();
			break;
		default:
			qDebug() << "Unknown option selected.";
			break;
		} });
}

//创建棋盘格标定工具栏
void StudyCorr::ChessToolBar()
{
	//添加棋盘格标定工具栏
	chessToolBar = new QToolBar(this);//工具栏默认不放在树中
	//addToolBar(ToolBar)
	addToolBar(Qt::TopToolBarArea, chessToolBar);//更改停靠区域
	//设置允许停靠区域
	chessToolBar->setAllowedAreas(Qt::TopToolBarArea);
	//工具栏加工具项
	QComboBox* ComboBox = new QComboBox(this);//添加下拉框
	//QLabel* ComBoxLable = new QLabel("相机个数");
	ComboBox->setToolTip("相机个数");

	// 向下拉框中添加选项 
	ComboBox->addItem("2");
	//ComboBox->addItem("3");
	ComboBox->addItem("4");
	// 将下拉框添加到工具栏
	chessToolBar->addWidget(ComboBox);
	chessToolBar->addSeparator();

	// 创建选择棋盘格边长的数字输入框
	this-> squareSizeSpinBox = new QSpinBox(this);
	squareSizeSpinBox->setMinimum(0); // 设置最小值为1（棋盘格边长不能小于1）
	squareSizeSpinBox->setMaximum(100000); // 设置最大值
	squareSizeSpinBox->setValue(3); // 默认边长为3
	squareSizeSpinBox->setSuffix("mm"); // 显示单位
	squareSizeSpinBox->setToolTip("输入棋盘格的边长（单位：毫米）");
	squareSizeSpinBox->setFixedWidth(110);
	chessToolBar->addWidget(squareSizeSpinBox);
	chessToolBar->addSeparator();

	// 创建选择棋盘格的行数和列数
	this->rowsSpinBox = new QSpinBox(this);
	rowsSpinBox->setMinimum(1);
	rowsSpinBox->setMaximum(1000);
	rowsSpinBox->setValue(12); // 默认行数
	rowsSpinBox->setSuffix("行");
	rowsSpinBox->setToolTip("输入棋盘格的角点行数");
	rowsSpinBox->setFixedWidth(100);

	this->colsSpinBox = new QSpinBox(this);
	colsSpinBox->setMinimum(1);
	colsSpinBox->setMaximum(1000);
	colsSpinBox->setValue(11); // 默认列数
	colsSpinBox->setSuffix("列");
	colsSpinBox->setToolTip("输入棋盘格的角点列数");
	colsSpinBox->setFixedWidth(100);

	// 将行数和列数添加到工具栏
	chessToolBar->addWidget(rowsSpinBox);
	chessToolBar->addSeparator();
	chessToolBar->addWidget(colsSpinBox);
	chessToolBar->addSeparator();

	StartCalibrationButton = chessToolBar->addAction("开始标定");
	StartCalibrationButton->setIcon(QIcon(":/icon/9.png"));

	connect(ComputeButton, &QAction::triggered, [=]()
	{
		chessToolBar->hide();
		computeToolBar->show();
	});
	//click start calibration button then start calibration
	connect(StartCalibrationButton, &QAction::triggered,this,&StudyCorr::ChessCalibrationButtonClicked);

}

//****************************************************计算****************************************************//
//点击计算按钮
void StudyCorr::ComputeButtonClicked(int ComputeIndex)
{
	qDebug() << "Button clicked, opening dialog";

	ComputeDialog* dialog = new ComputeDialog(this, ComputeIndex);

	// 点击加载文件
	connect(dialog->TableWidget->horizontalHeader(), &QHeaderView::sectionClicked, dialog, &ComputeDialog::OnHeaderClicked);

	// Ensure dialog gets deleted when closed
	dialog->setAttribute(Qt::WA_DeleteOnClose);

	// Connect accepted signal to onOkClicked slot
	connect(dialog, &ComputeDialog::accepted,  [=]() {
		this->ComputeOKButtonClicked();
		QPixmap ref_img = QPixmap(QString(LeftComputeFilePath[0])); // 参考图像
		//QPixmap tar_img = QPixmap(QString(RightComputeFilePath[currentFrameIndex])); // 目标图像
		qDebug() << "ref_img: " << LeftComputeFilePath;
		displayImages(ref_img);

	});

	computeDialog = dialog;

	// Show the dialog
	dialog->show();

	// 手动更新主界面->你到底更不更新啊，还闪退，玩不起别玩
	this->update();
}

//点击ComputeDialog中的确定按钮
void StudyCorr::ComputeOKButtonClicked()
{
	qDebug() << "Dialog accepted, running onOkClicked";
	if (computeDialog) {
		LeftComputeFilePath.clear();
		RightComputeFilePath.clear();
		LeftComputeFilePath = computeDialog->GetLeftFilePath();
		RightComputeFilePath = computeDialog->GetRightFilePath();// 获取文件名列表
		qDebug() << "LeftComputeFilePath: " << LeftComputeFilePath;
		computeImageFiles[ComputeIndex] = qMakePair(LeftComputeFilePath, RightComputeFilePath);

		// 添加顶层项
		QTreeWidgetItem* ComputeItem = new QTreeWidgetItem(QStringList(computeDialog->NameLineEdit->text()));
		TreeWidget2->addTopLevelItem(ComputeItem);
		ComputeItem->setTextAlignment(0, Qt::AlignLeft);// 设置列文本居中

		for (int row = 0; row < computeDialog->TableWidget->rowCount(); ++row) {
			QTableWidgetItem* item = computeDialog->TableWidget->item(row, 1);
			if (item) {
				// 获取文件名
				QString FileName = item->text();

				// 添加子节点
				QTreeWidgetItem* ComputeImage = new QTreeWidgetItem(QStringList(FileName));
				ComputeItem->addChild(ComputeImage);


				// 设置子项文本居中
				ComputeImage->setTextAlignment(0, Qt::AlignLeft);
			}
		}
		TreeWidget1->update();  // 强制刷新 QTreeWidget
	}
}

//创建计算工具栏
void StudyCorr::ComputeToolBar()
{
	// 创建工具栏并设置停靠区域
	this->computeToolBar = new QToolBar(this);
	addToolBar(Qt::TopToolBarArea, computeToolBar);
	computeToolBar->setAllowedAreas(Qt::TopToolBarArea);
	// 创建按钮并连接到相应的功能
	this->rectAction = new QAction("矩形", this);
	this->circleAction = new QAction("圆形", this);
	this->polygonAction = new QAction("多边形", this);
	this->cropPolygonAction = new QAction("裁剪多边形", this);
	this->dragROIAction = new QAction("拖动ROI", this);
	this->deleteAction = new QAction("删除", this);
	this->seedPoints = new QAction("种子点", this);
	this->autoROI = new QAction("自动选取ROI", this);
	// 创建步长的数字输入框
	this->stepSizeSpinBox = new QSpinBox(this);
	stepSizeSpinBox->setMinimum(0); // 设置最小值为0
	stepSizeSpinBox->setMaximum(1000); // 设置最大值
	stepSizeSpinBox->setValue(5); // 默认步长为5
	stepSizeSpinBox->setSuffix("pixel"); // 显示单位
	stepSizeSpinBox->setToolTip("输入步长（单位：像素）");
	stepSizeSpinBox->setFixedWidth(110);
	// 创建子集大小的数字输入框
	this->subSizeSpinBox = new QSpinBox(this);
	subSizeSpinBox->setMinimum(0); // 设置最小值为0
	subSizeSpinBox->setMaximum(100); // 设置最大值
	subSizeSpinBox->setValue(30); // 默认子集大小为30
	subSizeSpinBox->setSuffix("pixel"); // 显示单位
	subSizeSpinBox->setToolTip("输入子集大小（单位：像素）");
	subSizeSpinBox->setFixedWidth(110);
	// 创建开始计算按钮
	this->StartComputeButton = new QAction("开始计算", this);

	computeToolBar->addAction(rectAction);
	computeToolBar->addAction(circleAction);
	computeToolBar->addAction(polygonAction);
	computeToolBar->addAction(cropPolygonAction);
	computeToolBar->addAction(dragROIAction);
	computeToolBar->addAction(deleteAction);
	computeToolBar->addAction(seedPoints);
	computeToolBar->addAction(autoROI);
	computeToolBar->addSeparator();
	computeToolBar->addWidget(stepSizeSpinBox);
	computeToolBar->addSeparator();
	computeToolBar->addWidget(subSizeSpinBox);
	computeToolBar->addSeparator();
	computeToolBar->addAction(StartComputeButton);

	// 连接按钮到设置绘制模式,setAcceptHoverEvents(true)：接受鼠标悬停事件
	connect(rectAction, &QAction::triggered, this, [=]() {
		drawable1->setDrawMode(Drawable::Rectangle);
		view1->viewport()->update();
		});
	connect(circleAction, &QAction::triggered, this, [=]() {
		drawable1->setDrawMode(Drawable::Circle);
		view1->viewport()->update();
		});
	connect(polygonAction, &QAction::triggered, this, [=]() {
		drawable1->setDrawMode(Drawable::Polygon);
		view1->viewport()->update();
		});
	connect(cropPolygonAction, &QAction::triggered, this, [=]() {
		drawable1->setDrawMode(Drawable::ClipPolygon);
		view1->viewport()->update();
		});
	connect(dragROIAction, &QAction::triggered, this, [=]() {
		drawable1->setDrawMode(Drawable::Drag);
		view1->viewport()->update();
		});
	connect(deleteAction, &QAction::triggered, this, [=]() {
		drawable1->setDrawMode(Drawable::Delete);
		view1->viewport()->update();
		});

	connect(stepSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &StudyCorr::updateROICalculationPoints);
	connect(subSizeSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &StudyCorr::updateROICalculationPoints);
	connect(StartComputeButton, &QAction::triggered, this, [=]() {
		this->DicComputePOIQueue2D(dicConfig);
	});

	connect(CalibrationButton, &QAction::triggered, [=]() {
		computeToolBar->hide();
		chessToolBar->show(); }
	);

}

//****************************************************图像显示****************************************************//

//显示单幅图像
void StudyCorr::displayImages(const  cv::Mat& img)
{
	if (!item1) {
		qDebug() << "item1 ！";
		return;
	}

	if (img.empty()) {
		qDebug() << "图像为空，跳过。";
		return;
	}

	QImage img_qimg = cvMatToQImage(img);
	if (img_qimg.isNull()) {
		qDebug() << "图像转换失败，跳过。";
		return;
	}
	item1->setPixmap(QPixmap::fromImage(img_qimg));
	// 调整视图缩放并居中
	view1->fitInView(item1, Qt::KeepAspectRatio);
	view2->fitInView(item2, Qt::KeepAspectRatio);
}

void StudyCorr::displayImages(const  QPixmap& img1)
{
	if (!item1) {
		qDebug() << "item1 ！";
		return;
	}

	item1->setPixmap(img1);
	// 调整视图缩放并居中
	view1->fitInView(item1, Qt::KeepAspectRatio);
	view2->fitInView(item2, Qt::KeepAspectRatio);
}

//显示左右两幅图像
void StudyCorr::displayImages(const  cv::Mat& img1, const  cv::Mat& img2)
{
	if (!item1) {
		qDebug() << "item1 ！";
		return;
	}

	QImage img1_qimg1 = cvMatToQImage(img1);
	QImage img2_qimg2 = cvMatToQImage(img2);
	if (img1_qimg1.isNull() || img2_qimg2.isNull()) {
		qDebug() << "图像转换失败，跳过。";
		return;
	}

	// ... 加载QPixmap（不缩放）
	item1->setPixmap(QPixmap::fromImage(img1_qimg1));
	item2->setPixmap(QPixmap::fromImage(img2_qimg2));

	// 调整视图缩放并居中
	view1->fitInView(item1, Qt::KeepAspectRatio);
	view2->fitInView(item2, Qt::KeepAspectRatio);
	view1->setAlignment(Qt::AlignCenter);
	view2->setAlignment(Qt::AlignCenter);
}

void StudyCorr::displayImages(const QPixmap& img1, const QPixmap& img2)
{
	// ... 加载QPixmap
	item1->setPixmap(img1);
	item2->setPixmap(img2);

	// 调整视图缩放并居中
	view1->fitInView(item1, Qt::KeepAspectRatio);
	view2->fitInView(item2, Qt::KeepAspectRatio);
	view1->setAlignment(Qt::AlignCenter);
	view2->setAlignment(Qt::AlignCenter);
}

//CV::Mat转QImage
QImage StudyCorr::cvMatToQImage(const cv::Mat& mat)
{
	if (mat.empty()) return QImage();

	switch (mat.type()) {
	case CV_8UC1:  // 灰度图
		return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8).copy();
	case CV_8UC3:  // BGR 彩色图
	{
		cv::Mat rgbMat;
		cv::cvtColor(mat, rgbMat, cv::COLOR_BGR2RGB);
		return QImage(rgbMat.data, rgbMat.cols, rgbMat.rows, rgbMat.step, QImage::Format_RGB888).copy();
	}
	case CV_8UC4:  // 带透明通道
		return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32).copy();
	default:
		qDebug() << "不支持的图像类型";
		return QImage();
	}
}

void StudyCorr::ChessCalibrationButtonClicked()
{
	calibrationFactory = new Calibrationfactory();
	calibrationFactory->setModel(Calibrationfactory::CalibrationModel::Chessboard, rowsSpinBox->value(), colsSpinBox->value(), squareSizeSpinBox->value(), LeftCameraFilePath, RightCameraFilePath);
	chessCalibration = dynamic_cast<ChessCalibration*>(calibrationFactory->getCalibration());
	bool prefare = chessCalibration->prefareStereoCalibration();
	if (!prefare)
	{
		std::cerr << "not prefared." << std::endl;
		return;
	}
	else
	{
		chessCalibration->startStereoCalibration();
	}

	// 避免重复创建定时器
	if (!timer) {
		timer = new QTimer(this);
		connect(timer, &QTimer::timeout, [=]() mutable
			{
				if (currentFrameIndex < chessCalibration->img1_frames.size()) {
					// 显示图像帧
					this->displayImages(chessCalibration->img1_frames[currentFrameIndex], chessCalibration->img2_frames[currentFrameIndex]);

					// 显示图像文件名
					QFileInfo fileInfo1(LeftCameraFilePath[currentFrameIndex]);
					QFileInfo fileInfo2(RightCameraFilePath[currentFrameIndex]);
					QString img1Name = fileInfo1.fileName();
					QString img2Name = fileInfo2.fileName();
					// 设置文字内容
					img1TextItem->setPlainText(img1Name);
					img2TextItem->setPlainText(img2Name);

					// 设置文字位置（图像下方）
					img1TextItem->setPos(240, view1->height() - 300);
					img2TextItem->setPos(240, view2->height() - 300);
					currentFrameIndex++;
				}
				else {
					timer->stop();  // 所有图像显示完成，停止定时器
					currentFrameIndex = 0;  // 重置索引，方便下次播放
					qDebug() << "所有图像已显示完成，停止计时器。";
				}
			});
	}

	if (!timer->isActive()) {
		currentFrameIndex = 0;  // 每次点击前重置索引
		timer->start(100);  // 每 500 毫秒更新一张图像
	}
}

//****************************************************ROI****************************************************//
// 更新ROI计算点
void StudyCorr::updateROICalculationPoints()
{
	int stepSize = stepSizeSpinBox->value();
	int subSize = subSizeSpinBox->value();
	drawable1->updateCalculationPoints(stepSize, subSize);
	view1->viewport()->update();
	dicConfig.LeftComputeFilePath = LeftComputeFilePath;
	dicConfig.RightComputeFilePath = RightComputeFilePath;
	poi_queue_L.clear();poi_queue_R.clear();
	poi_queue_L.resize(dicConfig.LeftComputeFilePath.size());
	poi_queue_R.resize(dicConfig.LeftComputeFilePath.size());
	poi_queue_L[0] = drawable1->getPOI2DQueue();
	poi_queue_R[0] = drawable1->getPOI2DQueue();
	qDebug() << "updateROICalculationPoints: POI count:" << poi_queue_L[0].size();
	#pragma omp parallel for
	for (int i = 1; i < dicConfig.LeftComputeFilePath.size(); ++i) {
		poi_queue_L[i] = poi_queue_L[0]; // 初始化其他帧的POI队列
		poi_queue_R[i] = poi_queue_R[0]; // 初始化其他帧的POI队列
	}
}

void StudyCorr::DicComputePOIQueue2D(DICconfig& dicConfig)
{
	dicConfig.stepSize = stepSizeSpinBox->value();
	dicConfig.subSize = subSizeSpinBox->value();

	opencorr::Image2D ref_img(dicConfig.LeftComputeFilePath[0].toStdString()); // 参考图像
	double timer_tic = omp_get_wtime();

	#pragma omp parallel for
	for (int i = 0; i < dicConfig.LeftComputeFilePath.size(); ++i) 
	{
		opencorr::Image2D tar_img(dicConfig.LeftComputeFilePath[i].toStdString()); // 目标图像
		opencorr::SIFT2D* sift = new opencorr::SIFT2D();
		sift->setImages(ref_img, tar_img);
		sift->prepare();
		sift->compute();
		opencorr::FeatureAffine2D* feature_affine = new opencorr::FeatureAffine2D(dicConfig.subSize/2, dicConfig.subSize/2, dicConfig.cpu_thread_number);
		feature_affine->setImages(ref_img, tar_img);
		feature_affine->setKeypointPair(sift->ref_matched_kp, sift->tar_matched_kp);
		feature_affine->prepare();
		feature_affine->compute(poi_queue_L[i]);

		opencorr::ICGN2D2 icgn2(dicConfig.subSize/2, dicConfig.subSize/2, 
			dicConfig.max_deformation_norm, dicConfig.max_iteration, dicConfig.cpu_thread_number);
		icgn2.setImages(ref_img, tar_img);
		icgn2.prepare();
		icgn2.compute(poi_queue_L[i]);
	}
	double timer_toc = omp_get_wtime();
	double consumed_time = timer_toc - timer_tic;
	qDebug() << "计算耗时：" << consumed_time << "秒";
	qDebug() << "计算完成。";
}

// void StudyCorr::DicComputePOIQueue2DS(DICconfig &dicConfig)
// {
// 	dicConfig.stepSize = stepSizeSpinBox->value();
// 	dicConfig.subSize = subSizeSpinBox->value();

// 	// 1. 加载参考帧左右图像
// 	opencorr::Image2D ref_img(dicConfig.LeftComputeFilePath[0].toStdString());
// 	opencorr::Image2D ref_img_r(dicConfig.RightComputeFilePath[0].toStdString());

// 	// 2. 设置相机参数（应从标定结果获取，这里为示例硬编码）
// 	opencorr::CameraIntrinsics view1_cam_intrinsics, view2_cam_intrinsics;
// 	opencorr::CameraExtrinsics view1_cam_extrinsics, view2_cam_extrinsics;
// 	view1_cam_intrinsics.fx = 6673.315918f;
// 	view1_cam_intrinsics.fy = 6669.302734f;
// 	view1_cam_intrinsics.fs = 0.f;
// 	view1_cam_intrinsics.cx = 872.15778f;
// 	view1_cam_intrinsics.cy = 579.95532f;
// 	view1_cam_intrinsics.k1 = 0.032258954f;
// 	view1_cam_intrinsics.k2 = -1.01141417f;
// 	view1_cam_intrinsics.k3 = 29.78838921f;
// 	view1_cam_intrinsics.k4 = 0;
// 	view1_cam_intrinsics.k5 = 0;
// 	view1_cam_intrinsics.k6 = 0;
// 	view1_cam_intrinsics.p1 = 0;
// 	view1_cam_intrinsics.p2 = 0;
// 	view1_cam_extrinsics.tx = 0;
// 	view1_cam_extrinsics.ty = 0;
// 	view1_cam_extrinsics.tz = 0;
// 	view1_cam_extrinsics.rx = 0;
// 	view1_cam_extrinsics.ry = 0;
// 	view1_cam_extrinsics.rz = 0;

// 	view2_cam_intrinsics.fx = 6607.618164f;
// 	view2_cam_intrinsics.fy = 6602.857422f;
// 	view2_cam_intrinsics.fs = 0.f;
// 	view2_cam_intrinsics.cx = 917.9733887f;
// 	view2_cam_intrinsics.cy = 531.6352539f;
// 	view2_cam_intrinsics.k1 = 0.064598486f;
// 	view2_cam_intrinsics.k2 = -4.531373978f;
// 	view2_cam_intrinsics.k3 = 29.78838921f;
// 	view2_cam_intrinsics.k4 = 0;
// 	view2_cam_intrinsics.k5 = 0;
// 	view2_cam_intrinsics.k6 = 0;
// 	view2_cam_intrinsics.p1 = 0;
// 	view2_cam_intrinsics.p2 = 0;
// 	view2_cam_extrinsics.tx = 122.24886f;
// 	view2_cam_extrinsics.ty = 1.8488892f;
// 	view2_cam_extrinsics.tz = 17.624638f;
// 	view2_cam_extrinsics.rx = 0.00307711f;
// 	view2_cam_extrinsics.ry = -0.33278773f;
// 	view2_cam_extrinsics.rz = 0.00524556f;

// 	opencorr::Calibration cam_view1_calib(view1_cam_intrinsics, view1_cam_extrinsics);
// 	opencorr::Calibration cam_view2_calib(view2_cam_intrinsics, view2_cam_extrinsics);
// 	cam_view1_calib.prepare(ref_img.height, ref_img.width);
// 	cam_view2_calib.prepare(ref_img_r.height, ref_img_r.width);

// 	int cpu_thread_number = dicConfig.cpu_thread_number > 0 ? dicConfig.cpu_thread_number : omp_get_num_procs() - 1;
// 	omp_set_num_threads(cpu_thread_number);

// 	opencorr::Stereovision stereo_reconstruction(&cam_view1_calib, &cam_view2_calib, cpu_thread_number);

// 	double timer_tic = omp_get_wtime();

// 	// 遍历所有帧，进行3D-DIC
// 	#pragma omp parallel for
// 	for (int i = 0; i < dicConfig.LeftComputeFilePath.size(); ++i)
// 	{
// 		opencorr::Image2D tar_img(dicConfig.LeftComputeFilePath[i].toStdString());
// 		opencorr::Image2D tar_img_r(dicConfig.RightComputeFilePath[i].toStdString());

// 		// 1) 左图时序匹配（SIFT+仿射+ICGN1）
// 		opencorr::SIFT2D sift_l;
// 		sift_l.setImages(ref_img, tar_img);
// 		sift_l.prepare();
// 		sift_l.compute();

// 		opencorr::FeatureAffine2D feature_affine_l(dicConfig.subSize/2, dicConfig.subSize/2, cpu_thread_number);
// 		feature_affine_l.setImages(ref_img, tar_img);
// 		feature_affine_l.setKeypointPair(sift_l.ref_matched_kp, sift_l.tar_matched_kp);
// 		feature_affine_l.prepare();
// 		feature_affine_l.compute(poi_queue_L[i]);

// 		opencorr::ICGN2D1 icgn1_l(dicConfig.subSize/2, dicConfig.subSize/2, 
// 			dicConfig.max_deformation_norm, dicConfig.max_iteration, cpu_thread_number);
// 		icgn1_l.setImages(ref_img, tar_img);
// 		icgn1_l.prepare();
// 		icgn1_l.compute(poi_queue_L[i]);

// 		// 2) 右图时序匹配（SIFT+仿射+ICGN1）
// 		opencorr::SIFT2D sift_r;
// 		sift_r.setImages(ref_img_r, tar_img_r);
// 		sift_r.prepare();
// 		sift_r.compute();

// 		opencorr::FeatureAffine2D feature_affine_r(dicConfig.subSize/2, dicConfig.subSize/2, cpu_thread_number);
// 		feature_affine_r.setImages(ref_img_r, tar_img_r);
// 		feature_affine_r.setKeypointPair(sift_r.ref_matched_kp, sift_r.tar_matched_kp);
// 		feature_affine_r.prepare();
// 		feature_affine_r.compute(poi_queue_R[i]);

// 		opencorr::ICGN2D1 icgn1_r(dicConfig.subSize/2, dicConfig.subSize/2, 
// 			dicConfig.max_deformation_norm, dicConfig.max_iteration, cpu_thread_number);
// 		icgn1_r.setImages(ref_img_r, tar_img_r);
// 		icgn1_r.prepare();
// 		icgn1_r.compute(poi_queue_R[i]);

// 		// 3) 立体重建
// 		std::vector<opencorr::Point2D> ref_view1_pt, ref_view2_pt, tar_view1_pt, tar_view2_pt;
// 		std::vector<opencorr::Point3D> ref_pt_3d, tar_pt_3d;
// 		ref_view1_pt.reserve(poi_queue_L[i].size());
// 		ref_view2_pt.reserve(poi_queue_R[i].size());
// 		tar_view1_pt.reserve(poi_queue_L[i].size());
// 		tar_view2_pt.reserve(poi_queue_R[i].size());
// 		ref_pt_3d.resize(poi_queue_L[i].size());
// 		tar_pt_3d.resize(poi_queue_L[i].size());

// 		for (size_t j = 0; j < poi_queue_L[i].size(); ++j) {
// 			// 参考帧左右视点
// 			ref_view1_pt.push_back(opencorr::Point2D(poi_queue_L[i][j].x, poi_queue_L[i][j].y));
// 			ref_view2_pt.push_back(opencorr::Point2D(poi_queue_R[i][j].x, poi_queue_R[i][j].y));
// 			// 目标帧左右视点
// 			tar_view1_pt.push_back(opencorr::Point2D(
// 				poi_queue_L[i][j].x + poi_queue_L[i][j].deformation.u,
// 				poi_queue_L[i][j].y + poi_queue_L[i][j].deformation.v));
// 			tar_view2_pt.push_back(opencorr::Point2D(
// 				poi_queue_R[i][j].x + poi_queue_R[i][j].deformation.u,
// 				poi_queue_R[i][j].y + poi_queue_R[i][j].deformation.v));
// 		}

// 		stereo_reconstruction.prepare();
// 		stereo_reconstruction.reconstruct(ref_view1_pt, ref_view2_pt, ref_pt_3d);
// 		stereo_reconstruction.reconstruct(tar_view1_pt, tar_view2_pt, tar_pt_3d);

// 		// 保存3D结果到poi_queue_2DS
// 		for (size_t j = 0; j < poi_queue_R[i].size(); ++j) {
// 			poi_queue_R[i][j].ref_coor = ref_pt_3d[j];
// 			poi_queue_R[i][j].tar_coor = tar_pt_3d[j];
// 			poi_queue_R[i][j].deformation.u = tar_pt_3d[j].x - ref_pt_3d[j].x;
// 			poi_queue_R[i][j].deformation.v = tar_pt_3d[j].y - ref_pt_3d[j].y;
// 			poi_queue_R[i][j].deformation.w = tar_pt_3d[j].z - ref_pt_3d[j].z;
// 		}
// 	}

// 	double timer_toc = omp_get_wtime();
// 	double consumed_time = timer_toc - timer_tic;
// 	qDebug() << "3D-DIC计算耗时：" << consumed_time << "秒";
// 	qDebug() << "3D-DIC计算完成。";
// }


void StudyCorr::downloadData()	
{
	// 使用 QFileDialog 选择文件夹路径
	QString folder = QFileDialog::getExistingDirectory(this, "选择数据下载目录");

	if (!folder.isEmpty()) {
		QDir dir(folder);

		// 检查目录是否存在，如果不存在则创建
		if (!dir.exists()) {
			if (!dir.mkpath(folder)) {
				QMessageBox::critical(this, "错误", "无法创建目录！");
				return;
			}
		}

		// 在这里添加下载数据的逻辑
		opencorr::Image2D ref_img(dicConfig.LeftComputeFilePath[0].toStdString());
		std::string delimiter = ",";
		int ref_height = ref_img.height;
		int ref_width = ref_img.width;

		#pragma omp parallel for
		for (int i = 0; i < dicConfig.LeftComputeFilePath.size(); ++i)
		{
			std::string tempfile_path = dicConfig.LeftComputeFilePath[i].toStdString();
			// 获取文件名（不带路径）
			QString qfileName = QFileInfo(QString::fromStdString(tempfile_path)).baseName();
			// 拼接保存路径
			QString savePath = QDir(folder).filePath(qfileName + ".csv");
			std::string file_path = savePath.toStdString();

			opencorr::IO2D in_out;
			in_out.setDelimiter(delimiter);
			in_out.setHeight(ref_height);
			in_out.setWidth(ref_width);
			in_out.setPath(file_path);
			in_out.saveTable2D(poi_queue_L[i]);
		}
		QMessageBox::information(this, "成功", "数据下载成功！");
	}
	else {
		QMessageBox::warning(this, "警告", "未选择任何目录！");
	}
}
