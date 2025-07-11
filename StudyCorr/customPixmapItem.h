#pragma once
#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QGraphicsView>
#include "Drawable.h"

class CustomPixmapItem : public QGraphicsPixmapItem {
public:
    explicit CustomPixmapItem(QGraphicsItem* parent = nullptr);
    void setDrawable(Drawable* drawable);
    Drawable* getDrawable() const { return drawable; }

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
    void hoverMoveEvent(QGraphicsSceneHoverEvent* event) override;
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
    void wheelEvent(QGraphicsSceneWheelEvent* event) override; 
private:
    Drawable* drawable = nullptr;
};