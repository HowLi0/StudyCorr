#include "customPixmapItem.h"

CustomPixmapItem::CustomPixmapItem(QGraphicsItem* parent)
    : QGraphicsPixmapItem(parent), drawable(nullptr) {
    setAcceptHoverEvents(true); // 必须启用悬停事件
    setFlag(QGraphicsItem::ItemIsFocusable);  // 必须启用焦点
    setFocus();                                // 确保Item能接收键盘/滚轮事件
}

void CustomPixmapItem::setDrawable(Drawable* drawable) {
    this->drawable = drawable;
}


void CustomPixmapItem::mousePressEvent(QGraphicsSceneMouseEvent* event) {
    if (drawable) {
        drawable->handleMousePress(event);
        update();
        event->accept();
    }
    else {
        QGraphicsPixmapItem::mousePressEvent(event);
    }
}

void CustomPixmapItem::hoverMoveEvent(QGraphicsSceneHoverEvent* event) {
    if (drawable) {
        qDebug() << "Hover moved: " << event->scenePos();
        drawable->handleMouseMove(mapToScene(event->pos()));
        update();
    }
    QGraphicsPixmapItem::hoverMoveEvent(event);
}

void CustomPixmapItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* event) {
    if (drawable) {
        drawable->handleMouseRelease(event);
        update();
        event->accept();
    }
    else {
        QGraphicsPixmapItem::mouseReleaseEvent(event);
    }
}

void CustomPixmapItem::wheelEvent(QGraphicsSceneWheelEvent* event) {
    // 获取关联的视图（假设Item仅在一个视图中显示）
    QList<QGraphicsView*> views = scene()->views();
    if (views.isEmpty()) return;
    QGraphicsView* view = views.first();

    // 计算缩放因子（向上滚动放大，向下滚动缩小）
    qreal zoomFactor = (event->delta() > 0) ? 1.1 : 0.9;

    // 获取鼠标在场景中的位置（缩放中心点）
    QPointF scenePos = mapToScene(event->pos());

    // 调整视图的变换矩阵以实现中心缩放
    view->setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    view->scale(zoomFactor, zoomFactor);

    // 阻止事件继续传递
    event->accept();
}