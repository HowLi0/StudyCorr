#include "Drawable.h"
#include "Shape.h"

Drawable::Drawable(QGraphicsPixmapItem* pixmapItem)
    : pixmapItem(pixmapItem), drawMode(None), m_stepSize(5), m_subSize(30) {
    // 初始化绘制器为 nullptr
    shapeDrawer = nullptr;
}

void Drawable::setDrawMode(DrawMode mode) {
    if (mode == drawMode) return;
    resetDrawing();
    drawMode = mode;
    switch (drawMode) {
    case None:
        shapeDrawer = nullptr;
        break;
    case Rectangle:
        shapeDrawer = new RectangleDrawer(pixmapItem);
        break;
    case Circle:
        shapeDrawer = new CircleDrawer(pixmapItem);
        break;
    case Polygon:
        shapeDrawer = new PolygonDrawer(pixmapItem);
        break;
    case ClipPolygon:  
        shapeDrawer = new ClipPolygonDrawer(pixmapItem);
        break;
    case Delete:      
        shapeDrawer = new DeleteDrawer(pixmapItem);
        break;
    case Drag:
        shapeDrawer = new DragDrawer(pixmapItem);
        break;
    default:
        shapeDrawer = nullptr;
        break;
    }
}

void Drawable::handleMousePress(QGraphicsSceneMouseEvent* event) {
    if (shapeDrawer) shapeDrawer->handleMousePress(event);
}

void Drawable::handleMouseMove(const QPointF& scenePos) {
    if (shapeDrawer) shapeDrawer->handleMouseMove(scenePos);
}

void Drawable::handleMouseRelease(QGraphicsSceneMouseEvent* event) {
    if (shapeDrawer) shapeDrawer->handleMouseRelease(event);
}

void Drawable::resetDrawing() {
    // 如果是拖拽绘制器，先清理状态
    if (shapeDrawer && drawMode == Drag) {
        DragDrawer* dragDrawer = dynamic_cast<DragDrawer*>(shapeDrawer);
        if (dragDrawer) {
            dragDrawer->cleanup();
        }
    }
    
    delete shapeDrawer;
    shapeDrawer = nullptr;
}

void Drawable::updateCalculationPoints(int stepSize, int subSize) {
    calculationPoints.clear();
    m_stepSize= stepSize;
    m_subSize = subSize;
    if (shapeDrawer) {
        shapeDrawer->updateCalculationPoints(m_stepSize, m_subSize);
    }

}

QVector<QPointF> Drawable::getCalculationPoints() const 
{
    return calculationPoints;
}

std::vector<opencorr::POI2D> Drawable::getPOI2DQueue()
{
    poi_queue_2D.clear();
    poi_queue_2D.resize(calculationPoints.size());
    #pragma omp parallel for
    for (int i = 0; i < calculationPoints.size(); i++)
    {
        poi_queue_2D[i] = opencorr::POI2D(static_cast<float>(calculationPoints[i].x()), static_cast<float>(calculationPoints[i].y()));
    }
    return poi_queue_2D;
}

std::vector<StudyCorr_GPU::CudaPOI2D> Drawable::getCudaPOI2DQueue()
{
    poi_queue_studycorr.clear();
    poi_queue_studycorr.resize(calculationPoints.size());
    #pragma omp parallel for
    for (int i = 0; i < calculationPoints.size(); i++)
    {
        StudyCorr_GPU::CudaPOI2D poi;
        poi.x = static_cast<float>(calculationPoints[i].x());
        poi.y = static_cast<float>(calculationPoints[i].y());
        poi_queue_studycorr[i] = poi;
    }
    return poi_queue_studycorr;
}
