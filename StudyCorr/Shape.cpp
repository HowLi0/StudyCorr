#include "shape.h"
#include "Drawable.h"
#include <QTimer>

cv::Mat QImageToCvMat(const QImage& image) {
    return cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.bits(), image.bytesPerLine()).clone();
}

QImage CvMatToQImage(const cv::Mat& mat) {
    return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32).copy();
}

void drawPointsOnImage(QPixmap& pixmap, const QVector<QPointF>& points) {
    cv::Mat img = QImageToCvMat(pixmap.toImage()); // QPixmap 转 cv::Mat

    for (const QPointF& point : points) {
        cv::circle(img, cv::Point(point.x(), point.y()), 1, cv::Scalar(0, 0, 255), -1); // 红色点
    }

    pixmap = QPixmap::fromImage(CvMatToQImage(img)); // cv::Mat 转 QPixmap
}

// 显示点5秒后自动清除
void showPointsTemporarily(QGraphicsPixmapItem* pixmapItem, const QPixmap& originalPixmap, const QVector<QPointF>& points) {
    QPixmap pixmap = originalPixmap;
    drawPointsOnImage(pixmap, points);
    pixmapItem->setPixmap(pixmap);
    pixmapItem->update();
    
    // 5秒后恢复原始图像
    QTimer::singleShot(2000, [pixmapItem, originalPixmap]() {
        pixmapItem->setPixmap(originalPixmap);
        pixmapItem->update();
    });
}

// 矩形绘制器实现
RectangleDrawer::RectangleDrawer(QGraphicsPixmapItem* parent) : pixmapItem(parent)
{
    originalPixmap = pixmapItem->pixmap();
}

void RectangleDrawer::handleMousePress(QGraphicsSceneMouseEvent* event) {
    if (event->button() != Qt::LeftButton) return;
    QPointF localPos = pixmapItem->mapFromScene(event->scenePos());

    if (!isDrawing) {
        startPoint = localPos;
        isDrawing = true;
        if (!previewRect) {
            previewRect = new QGraphicsRectItem(pixmapItem);
            previewRect->setPen(QPen(Qt::green, 2, Qt::DotLine));
        }
        previewRect->setRect(QRectF(startPoint, QSizeF(2, 2)));
    }
    else {
        if (previewRect) {
            QGraphicsRectItem* finalRect = new QGraphicsRectItem(previewRect->rect(), pixmapItem);
            finalRect->setPen(QPen(Qt::red, 2));
            finalRects.append(finalRect); // 添加到 QVector 中
            delete previewRect;
            previewRect = nullptr;
        }
        isDrawing = false;
    }
}

void RectangleDrawer::handleMouseMove(const QPointF& scenePos) {
    if (!isDrawing || !previewRect) return;
    QRectF rect(startPoint, pixmapItem->mapFromScene(scenePos));
    rect = rect.intersected(pixmapItem->boundingRect());
    previewRect->setRect(rect.normalized());
}

void RectangleDrawer::handleMouseRelease(QGraphicsSceneMouseEvent* event) {
    CustomPixmapItem* customPixmapItem = dynamic_cast<CustomPixmapItem*>(pixmapItem);
    if (customPixmapItem) {
        Drawable* drawable = customPixmapItem->getDrawable();
        if (drawable) {
            drawable->updateCalculationPoints(drawable->m_stepSize, drawable->m_subSize);
        }
    }
}

void RectangleDrawer::updateCalculationPoints(int stepSize, int subSize) {
    if (finalRects.isEmpty()) return;

    pixmapItem->setPixmap(originalPixmap);
    pixmapItem->update();

    QPixmap pixmap = pixmapItem->pixmap();
    cv::Mat img = QImageToCvMat(pixmap.toImage());
    cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
    QVector<QPointF> calculationPoints;

    // 清空并提取所有ROI
    ROI.clear();
    for (QGraphicsRectItem* rect : finalRects) {
        // 考虑位置偏移
        QPointF offset = rect->pos();
        QRectF actualRect = rect->rect().translated(offset);
        
        // 在mask上绘制矩形
        cv::rectangle(mask,
            cv::Point(actualRect.left(), actualRect.top()),
            cv::Point(actualRect.right(), actualRect.bottom()),
            cv::Scalar(255), cv::FILLED);
        
        // 提取 ROI 时也考虑偏移
        QRect roiRect = actualRect.toRect().intersected(pixmap.rect());
        if (!roiRect.isEmpty()) {
            QImage roiImg = pixmap.toImage().copy(roiRect);
            cv::Mat mat = QImageToCvMat(roiImg);
            cv::Mat matBGR;
            cv::cvtColor(mat, matBGR, cv::COLOR_BGRA2BGR);
            ROI.append(matBGR.clone());
        }
    }

    // 计算ROI内的点
    for (int y = 0; y <= mask.rows - subSize; y += stepSize) {
        for (int x = 0; x <= mask.cols - subSize; x += stepSize) {
            cv::Rect window(x, y, subSize, subSize);
            if (cv::countNonZero(mask(window)) == subSize * subSize) {
                calculationPoints.append(QPointF(x + subSize / 2, y + subSize / 2));
            }
        }
    }
    CustomPixmapItem* customPixmapItem = dynamic_cast<CustomPixmapItem*>(pixmapItem);
    if (customPixmapItem) {
        Drawable* drawable = customPixmapItem->getDrawable();
        if (drawable) {
            drawable->calculationPoints = calculationPoints;
            qDebug() << "Updated calculation points in Drawable";
            drawable->ROI = ROI; // 更新ROI
        }
    }

    // 显示点
    showPointsTemporarily(pixmapItem, originalPixmap, calculationPoints);
    pixmapItem->update();
}


// 圆形绘制器实现
CircleDrawer::CircleDrawer(QGraphicsPixmapItem* parent) : pixmapItem(parent) {
    originalPixmap = pixmapItem->pixmap();
}

void CircleDrawer::handleMousePress(QGraphicsSceneMouseEvent* event) {
    if (event->button() != Qt::LeftButton) return;
    QPointF localPos = pixmapItem->mapFromScene(event->scenePos());
    circlePoints.append(localPos);

    if (circlePoints.size() == 3) {
        createFinalCircle();
        reset();
    }
}

void CircleDrawer::handleMouseMove(const QPointF& scenePos) {
    QPointF localPos = pixmapItem->mapFromScene(scenePos);
    if (circlePoints.size() == 1) {
        updateRadiusPreview(localPos);
    }
    else if (circlePoints.size() == 2) {
        updateCircumcirclePreview(localPos);
    }
}

void CircleDrawer::handleMouseRelease(QGraphicsSceneMouseEvent* event) {
    CustomPixmapItem* customPixmapItem = dynamic_cast<CustomPixmapItem*>(pixmapItem);
    if (customPixmapItem) {
        Drawable* drawable = customPixmapItem->getDrawable();
        if (drawable) {
            drawable->updateCalculationPoints(drawable->m_stepSize, drawable->m_subSize);
        }
    }
}


void CircleDrawer::updateCalculationPoints(int stepSize, int subSize) {
    if (finalCircles.isEmpty()) return;

    pixmapItem->setPixmap(originalPixmap);
    pixmapItem->update();

    QPixmap pixmap = pixmapItem->pixmap();
    cv::Mat img = QImageToCvMat(pixmap.toImage());
    cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
    QVector<QPointF> calculationPoints;

    // 清空并提取所有ROI
    ROI.clear();
    for (QGraphicsEllipseItem* circle : finalCircles) {
        // 考虑位置偏移
        QPointF offset = circle->pos();
        QRectF actualRect = circle->rect().translated(offset);
        
        // 在mask上绘制圆
        cv::Point center(actualRect.center().x(), actualRect.center().y());
        int radius = actualRect.width() / 2;
        cv::circle(mask, center, radius, cv::Scalar(255), cv::FILLED);
        
        // 提取 ROI 时也考虑偏移
        QRect roiRect = actualRect.toRect().intersected(pixmap.rect());
        if (!roiRect.isEmpty()) {
            QImage roiImg = pixmap.toImage().copy(roiRect);
            cv::Mat mat = QImageToCvMat(roiImg);
            cv::Mat matBGR;
            cv::cvtColor(mat, matBGR, cv::COLOR_BGRA2BGR);
            ROI.append(matBGR.clone());
        }
    }

    for (int y = 0; y <= mask.rows - subSize; y += stepSize) {
        for (int x = 0; x <= mask.cols - subSize; x += stepSize) {
            cv::Rect window(x, y, subSize, subSize);
            if (cv::countNonZero(mask(window)) == subSize * subSize) {
                calculationPoints.append(QPointF(x + subSize / 2, y + subSize / 2));
            }
        }
    }

    CustomPixmapItem* customPixmapItem = dynamic_cast<CustomPixmapItem*>(pixmapItem);
    if (customPixmapItem) {
        Drawable* drawable = customPixmapItem->getDrawable();
        if (drawable) {
            drawable->calculationPoints = calculationPoints;
            drawable->ROI = ROI; // 更新ROI
        }
    }

    showPointsTemporarily(pixmapItem, originalPixmap, calculationPoints);
    pixmapItem->update();
}


void CircleDrawer::updateRadiusPreview(const QPointF& endPos)
{
    if (!previewCircle) {
        previewCircle = new QGraphicsEllipseItem(pixmapItem);
        previewCircle->setPen(QPen(Qt::green, 2, Qt::DotLine));
    }
    qreal radius = QLineF(circlePoints[0], endPos).length();
    previewCircle->setRect(circlePoints[0].x() - radius, circlePoints[0].y() - radius, radius * 2, radius * 2);
}

void CircleDrawer::updateCircumcirclePreview(const QPointF& thirdPos)
{
    QPointF center;
    qreal radius;
    if (calculateCircumcircle(circlePoints[0], circlePoints[1], thirdPos, center, radius)) {
        if (!previewCircle) {
            previewCircle = new QGraphicsEllipseItem(pixmapItem);
            previewCircle->setPen(QPen(Qt::red, 2, Qt::DotLine));
        }
        previewCircle->setRect(center.x() - radius, center.y() - radius, radius * 2, radius * 2);
    }
}

void CircleDrawer::createFinalCircle()
{
    QPointF center;
    qreal radius;
    if (calculateCircumcircle(circlePoints[0], circlePoints[1], circlePoints[2], center, radius)) {
        QGraphicsEllipseItem* ellipseItem = new QGraphicsEllipseItem(center.x() - radius, center.y() - radius, radius * 2, radius * 2, pixmapItem);
        ellipseItem->setPen(QPen(Qt::red, 2));
        finalCircles.append(ellipseItem); // 添加到 QVector 中
    }
}

bool CircleDrawer::calculateCircumcircle(const QPointF& p1, const QPointF& p2, const QPointF& p3, QPointF& center, qreal& radius)
{
    qreal ax = p2.x() - p1.x();
    qreal ay = p2.y() - p1.y();
    qreal bx = p3.x() - p2.x();
    qreal by = p3.y() - p2.y();
    qreal det = ax * by - ay * bx;
    if (qFuzzyIsNull(det)) return false;
    center.setX((by * (ax * (p1.x() + p2.x()) + ay * (p1.y() + p2.y())) - ay * (bx * (p2.x() + p3.x()) + by * (p2.y() + p3.y()))) / (2 * det));
    center.setY((ax * (bx * (p2.x() + p3.x()) + by * (p2.y() + p3.y())) - bx * (ax * (p1.x() + p2.x()) + ay * (p1.y() + p2.y()))) / (2 * det));
    radius = QLineF(center, p1).length();
    return true;
}

void CircleDrawer::reset()
{
    circlePoints.clear();
    delete previewCircle;
    previewCircle = nullptr;
}

// 构造函数
PolygonDrawer::PolygonDrawer(QGraphicsPixmapItem* parent) : pixmapItem(parent) {
    originalPixmap = pixmapItem->pixmap();
}

void PolygonDrawer::handleMousePress(QGraphicsSceneMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        QPointF localPos = pixmapItem->mapFromScene(event->scenePos());
        currentPolygon.append(localPos);
        updatePolygonPreview(localPos);
    }
    else if (event->button() == Qt::RightButton && currentPolygon.size() > 2) {
        currentPolygon.append(currentPolygon.first()); // 闭合多边形
        finalizePolygon();
        reset();
    }
}

void PolygonDrawer::handleMouseMove(const QPointF& scenePos) {
    if (!currentPolygon.isEmpty()) {
        updatePolygonPreview(pixmapItem->mapFromScene(scenePos));
    }
}

void PolygonDrawer::handleMouseRelease(QGraphicsSceneMouseEvent* event) {
    CustomPixmapItem* customPixmapItem = dynamic_cast<CustomPixmapItem*>(pixmapItem);
    if (customPixmapItem) {
        Drawable* drawable = customPixmapItem->getDrawable();
        if (drawable) {
            drawable->updateCalculationPoints(drawable->m_stepSize, drawable->m_subSize);
        }
    }
}

void PolygonDrawer::updateCalculationPoints(int stepSize, int subSize) {
    if (finalPolygons.isEmpty()) return;

    pixmapItem->setPixmap(originalPixmap);
    pixmapItem->update();

    QPixmap pixmap = pixmapItem->pixmap();
    cv::Mat img = QImageToCvMat(pixmap.toImage());
    cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
    QVector<QPointF> calculationPoints;

    // 清空并提取所有ROI
    ROI.clear();
    for (QGraphicsPolygonItem* polygon : finalPolygons) {
        // 考虑位置偏移
        QPointF offset = polygon->pos();
        QPolygonF actualPoly = polygon->polygon().translated(offset);
        
        // 在mask上绘制多边形
        std::vector<cv::Point> polyPoints;
        for (const QPointF& p : actualPoly) {
            polyPoints.emplace_back(cv::Point(p.x(), p.y()));
        }
        const cv::Point* pts[] = { polyPoints.data() };
        int numPoints[] = { static_cast<int>(polyPoints.size()) };
        cv::fillPoly(mask, pts, numPoints, 1, cv::Scalar(255));
        
        // 提取 ROI 时也考虑偏移
        QRectF bbox = actualPoly.boundingRect();
        QRect roiRect = bbox.toRect().intersected(pixmap.rect());
        if (!roiRect.isEmpty()) {
            QImage roiImg = pixmap.toImage().copy(roiRect);
            cv::Mat mat = QImageToCvMat(roiImg);
            cv::Mat matBGR;
            cv::cvtColor(mat, matBGR, cv::COLOR_BGRA2BGR);
            ROI.append(matBGR.clone());
        }
    }

    for (int y = 0; y <= mask.rows - subSize; y += stepSize) {
        for (int x = 0; x <= mask.cols - subSize; x += stepSize) {
            cv::Rect window(x, y, subSize, subSize);
            if (cv::countNonZero(mask(window)) == subSize * subSize) {
                calculationPoints.append(QPointF(x + subSize / 2, y + subSize / 2));
            }
        }
    }

    CustomPixmapItem* customPixmapItem = dynamic_cast<CustomPixmapItem*>(pixmapItem);
    if (customPixmapItem) {
        Drawable* drawable = customPixmapItem->getDrawable();
        if (drawable) {
            drawable->calculationPoints = calculationPoints;
            drawable->ROI = ROI; // 更新ROI
        }
    }

    showPointsTemporarily(pixmapItem, originalPixmap, calculationPoints);
    pixmapItem->update();
}


void PolygonDrawer::updatePolygonPreview(const QPointF& scenePos) {
    if (!currentPolygonItem) {
        currentPolygonItem = new QGraphicsPolygonItem(pixmapItem);
        currentPolygonItem->setPen(QPen(Qt::green, 2, Qt::DotLine));
    }
    QPolygonF preview = currentPolygon;
    preview.append(scenePos);
    currentPolygonItem->setPolygon(preview);
}

void PolygonDrawer::finalizePolygon() {
    if (currentPolygon.size() < 3) return;

    QGraphicsPolygonItem* finalItem = new QGraphicsPolygonItem(currentPolygon, pixmapItem);
    finalItem->setPen(QPen(Qt::red, 2));
    finalPolygons.append(finalItem); // 添加到 QVector 中
}

void PolygonDrawer::reset() {
    currentPolygon.clear();
    delete currentPolygonItem;
    currentPolygonItem = nullptr;
}

ClipPolygonDrawer::ClipPolygonDrawer(QGraphicsPixmapItem* parent) : pixmapItem(parent) {
    originalPixmap = pixmapItem->pixmap();
}

void ClipPolygonDrawer::handleMousePress(QGraphicsSceneMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        QPointF localPos = pixmapItem->mapFromScene(event->scenePos());
        clipPolygon.append(localPos);
        updateClipPreview(localPos);
    }
    else if (event->button() == Qt::RightButton && clipPolygon.size() > 2) {
        finalizeClip();
        reset();
    }
}

void ClipPolygonDrawer::handleMouseMove(const QPointF& scenePos)
{
    if (!clipPolygon.isEmpty()) {
        updateClipPreview(pixmapItem->mapFromScene(scenePos));
    }
}

void ClipPolygonDrawer::handleMouseRelease(QGraphicsSceneMouseEvent* event)
{
    CustomPixmapItem* customPixmapItem = dynamic_cast<CustomPixmapItem*>(pixmapItem);
    if (customPixmapItem) {
        Drawable* drawable = customPixmapItem->getDrawable();
        if (drawable) {
            drawable->updateCalculationPoints(drawable->m_stepSize, drawable->m_subSize);
        }
    }
}

void ClipPolygonDrawer::updateCalculationPoints(int stepSize, int subSize) {
    if (finalClipPolygons.isEmpty()) return;

    pixmapItem->setPixmap(originalPixmap);
    pixmapItem->update();

    QPixmap pixmap = pixmapItem->pixmap();
    cv::Mat img = QImageToCvMat(pixmap.toImage());
    cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8UC1);
    QVector<QPointF> calculationPoints;

    // 清空并提取所有ROI
    ROI.clear();
    for (QGraphicsPathItem* polygon : finalClipPolygons) {
        // 考虑位置偏移
        QPointF offset = polygon->pos();
        
        // 在mask上绘制裁剪多边形
        QPainterPath path = polygon->path();
        QPolygonF poly = path.toFillPolygon().translated(offset); // 应用偏移
        
        std::vector<cv::Point> polyPoints;
        for (const QPointF& p : poly) {
            polyPoints.emplace_back(cv::Point(p.x(), p.y()));
        }
        const cv::Point* pts[] = { polyPoints.data() };
        int numPoints[] = { static_cast<int>(polyPoints.size()) };
        cv::fillPoly(mask, pts, numPoints, 1, cv::Scalar(255));
        
        // 提取 ROI 时也考虑偏移
        QRectF bbox = poly.boundingRect();
        QRect roiRect = bbox.toRect().intersected(pixmap.rect());
        if (!roiRect.isEmpty()) {
            QImage roiImg = pixmap.toImage().copy(roiRect);
            cv::Mat mat = QImageToCvMat(roiImg);
            cv::Mat matBGR;
            cv::cvtColor(mat, matBGR, cv::COLOR_BGRA2BGR);
            ROI.append(matBGR.clone());
        }
    }

    for (int y = 0; y <= mask.rows - subSize; y += stepSize) {
        for (int x = 0; x <= mask.cols - subSize; x += stepSize) {
            cv::Rect window(x, y, subSize, subSize);
            if (cv::countNonZero(mask(window)) == subSize * subSize) {
                calculationPoints.append(QPointF(x + subSize / 2, y + subSize / 2));
            }
        }
    }

    CustomPixmapItem* customPixmapItem = dynamic_cast<CustomPixmapItem*>(pixmapItem);
    if (customPixmapItem) {
        Drawable* drawable = customPixmapItem->getDrawable();
        if (drawable) {
            drawable->calculationPoints = calculationPoints;
            drawable->ROI = ROI; // 更新ROI
        }
    }

    showPointsTemporarily(pixmapItem, originalPixmap, calculationPoints);
    pixmapItem->update();
}

void ClipPolygonDrawer::updateClipPreview(const QPointF& scenePos) {
    if (!previewClipItem) {
        previewClipItem = new QGraphicsPolygonItem(pixmapItem);
        previewClipItem->setPen(QPen(Qt::blue, 2, Qt::DotLine));
    }
    QPolygonF preview = clipPolygon;
    preview.append(scenePos);
    previewClipItem->setPolygon(preview);
}

void ClipPolygonDrawer::finalizeClip() {
    // 创建裁剪路径
    QPainterPath clipPath;
    clipPath.addPolygon(clipPolygon);

    // 遍历所有子项
    for(QGraphicsItem * item : pixmapItem->childItems()) {
        // 跳过自身预览项和 pixmapItem
        if (item == previewClipItem || item == pixmapItem) continue;

        // 获取原始形状的路径
        QPainterPath subjectPath = item->shape();
        // 计算交集路径
        QPainterPath intersected = subjectPath.intersected(clipPath);

        if (!intersected.isEmpty()) {
            // 创建新的路径项并继承原样式
            QGraphicsPathItem* newItem = new QGraphicsPathItem(intersected, pixmapItem);
			finalClipPolygons.append(newItem);
            // 复制原项的样式
            if (auto shapeItem = dynamic_cast<QAbstractGraphicsShapeItem*>(item)) {
                newItem->setPen(shapeItem->pen());
                newItem->setBrush(shapeItem->brush());
            }

            // 删除原项
            delete item;
        }
    }
}

void ClipPolygonDrawer::reset() {
    clipPolygon.clear();
    delete previewClipItem;
    previewClipItem = nullptr;
}

DeleteDrawer::DeleteDrawer(QGraphicsPixmapItem* parent) : pixmapItem(parent) {}

void DeleteDrawer::handleMousePress(QGraphicsSceneMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        QGraphicsScene* scene = pixmapItem->scene();
        if (scene) {
            QGraphicsItem* clickedItem = scene->itemAt(event->scenePos(), QTransform());
            if (clickedItem && clickedItem != pixmapItem) {
                delete clickedItem;
            }
        }
        CustomPixmapItem* customPixmapItem = dynamic_cast<CustomPixmapItem*>(pixmapItem);
        Drawable* drawable = customPixmapItem->getDrawable();
        if (drawable) {
            QPointF clickPos = pixmapItem->mapFromScene(event->scenePos());
            drawable->calculationPoints.erase(
                std::remove_if(drawable->calculationPoints.begin(), drawable->calculationPoints.end(),
                    [&](const QPointF& p) { return QLineF(p, clickPos).length() < 5; }),
                drawable->calculationPoints.end());

            // 保存原始图像，用于之后恢复
            QPixmap originalPixmap = pixmapItem->pixmap().copy();
            
            // 显示点5秒后自动清除
            showPointsTemporarily(pixmapItem, originalPixmap, drawable->calculationPoints);
            pixmapItem->update();
        }
    }
}

// 拖拽绘制器
DragDrawer::DragDrawer(QGraphicsPixmapItem* parent) : pixmapItem(parent) 
{
    originalPixmap = pixmapItem->pixmap();
}

DragDrawer::~DragDrawer() {
    cleanup();
}

void DragDrawer::handleMousePress(QGraphicsSceneMouseEvent* event) {
    if (event->button() != Qt::LeftButton) return;

    // 第一次左键点击：选择要拖拽的图形项
    QGraphicsScene* scene = pixmapItem->scene();
    if (!scene) return;
    
    if (isclick ==0) {
        QGraphicsItem* clickedItem = scene->itemAt(event->scenePos(), QTransform());
        // 检查点击项是否为可拖拽类型
        if (clickedItem && clickedItem != pixmapItem &&
            (dynamic_cast<QGraphicsRectItem*>(clickedItem) ||
             dynamic_cast<QGraphicsEllipseItem*>(clickedItem) ||
             dynamic_cast<QGraphicsPolygonItem*>(clickedItem) ||
             dynamic_cast<QGraphicsPathItem*>(clickedItem)))
        {
            draggedItem = clickedItem;
            dragStartPos = event->scenePos();
            itemStartPos = draggedItem->scenePos();
            isclick = 1; // 设置为第一次点击状态
            qDebug() << "开始拖拽";
            event->accept();
        }
    } 
    else if (isclick == 1) // 第二次左键点击
    {
        isclick = 0; // 重置点击状态
        qDebug() << "第二次左键点击 - 停止拖拽";
        if (draggedItem)
        {
            // 添加到最终拖拽项集合（去重）
            if (!finalDraggedItems.contains(draggedItem)) 
            {
                finalDraggedItems.append(draggedItem);
            }
            
        CustomPixmapItem* customPixmapItem = dynamic_cast<CustomPixmapItem*>(pixmapItem);
        Drawable* drawable = customPixmapItem->getDrawable();
            if (drawable) 
            {
                // 根据拖拽的图形类型，调用对应形状的更新方法
                if (dynamic_cast<QGraphicsRectItem*>(draggedItem)) {
                    // 临时创建矩形绘制器处理更新（使用现有图形数据）
                    RectangleDrawer tempDrawer(pixmapItem);
                    tempDrawer.finalRects = {dynamic_cast<QGraphicsRectItem*>(draggedItem)};
                    tempDrawer.updateCalculationPoints(drawable->m_stepSize, drawable->m_subSize);
                }
                else if (dynamic_cast<QGraphicsEllipseItem*>(draggedItem)) {
                    // 圆形处理
                    CircleDrawer tempDrawer(pixmapItem);
                    tempDrawer.finalCircles = {dynamic_cast<QGraphicsEllipseItem*>(draggedItem)};
                    tempDrawer.updateCalculationPoints(drawable->m_stepSize, drawable->m_subSize);
                }
                else if (dynamic_cast<QGraphicsPolygonItem*>(draggedItem)) {
                    // 多边形处理
                    PolygonDrawer tempDrawer(pixmapItem);
                    tempDrawer.finalPolygons = {dynamic_cast<QGraphicsPolygonItem*>(draggedItem)};
                    tempDrawer.updateCalculationPoints(drawable->m_stepSize, drawable->m_subSize);
                }
                else if (dynamic_cast<QGraphicsPathItem*>(draggedItem)) {
                    // 裁剪多边形处理
                    ClipPolygonDrawer tempDrawer(pixmapItem);
                    tempDrawer.finalClipPolygons = {dynamic_cast<QGraphicsPathItem*>(draggedItem)};
                    tempDrawer.updateCalculationPoints(drawable->m_stepSize, drawable->m_subSize);
                }

                // 强制刷新显示
                pixmapItem->update();
                if (pixmapItem->scene()) {
                    pixmapItem->scene()->update();
                }
            }
        draggedItem = nullptr;
    }
         event->accept(); 
    }
}

// 修复 handleMouseMove
void DragDrawer::handleMouseMove(const QPointF& scenePos) {
    // 仅在拖拽状态中才响应移动事件
    if (!draggedItem || isclick != 1) return; // 严格判断isclick为1

    QPointF delta = scenePos - dragStartPos;
    draggedItem->setPos(itemStartPos + delta);
    qDebug() << "拖拽中 - 位置:" << draggedItem->scenePos();
}

// 修复 handleMouseRelease
void DragDrawer::handleMouseRelease(QGraphicsSceneMouseEvent* event) 
{
    if (event->button() != Qt::LeftButton) return;
    // 仅在拖拽状态中记录释放位置，不修改isclick状态
    if (draggedItem && isclick == 1) {
        qDebug() << "鼠标释放 - 当前位置:" << draggedItem->scenePos();
    }
    event->accept();
}

void DragDrawer::cleanup() {
    draggedItem = nullptr;
    isclick = 0 ;// 确保状态重置
}

void DragDrawer::updateCalculationPoints(int stepSize, int subSize) 
{
    return; // 拖拽绘制器不需要更新计算点
}

