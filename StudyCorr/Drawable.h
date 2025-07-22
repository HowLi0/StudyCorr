#pragma once
#include <QObject>
#include <QGraphicsPixmapItem>
#include <opencv2/opencv.hpp>
#include"oc_poi.h"
#include"sc_poi.h"
#include <vector>
    
class  ShapeDrawer;

class Drawable : public QObject {
    Q_OBJECT
public:
    enum DrawMode { None, Rectangle, Circle, Polygon, ClipPolygon, Delete, Drag};
    explicit Drawable(QGraphicsPixmapItem* pixmapItem);
    void setDrawMode(DrawMode mode);
    void handleMousePress(QGraphicsSceneMouseEvent* event);
    void handleMouseMove(const QPointF& scenePos);
    void handleMouseRelease(QGraphicsSceneMouseEvent* event);
    void resetDrawing();
    void updateCalculationPoints(int stepSize, int subSize);
    QVector<QPointF> getCalculationPoints() const; // 添加方法获取计算点
    std::vector<opencorr::POI2D> getPOI2DQueue(); 
    std::vector<StudyCorr_GPU::CudaPOI2D> getCudaPOI2DQueue();
    QVector<cv::Mat> ROI; // 存储所有 ROI 的图像数据
    QVector<QPointF> calculationPoints; // 存储所有 ROI 中的计算点
    std::vector<opencorr::POI2D> poi_queue_2D; 
    std::vector<StudyCorr_GPU::CudaPOI2D> poi_queue_studycorr; // 存储所有的 POI
    int m_stepSize = 5, m_subSize = 30;
    ShapeDrawer* shapeDrawer = nullptr;
    void setPixmapItem(QGraphicsPixmapItem* item) { pixmapItem = item; }
private:
    QGraphicsPixmapItem* pixmapItem = nullptr;
    DrawMode drawMode = None;
};