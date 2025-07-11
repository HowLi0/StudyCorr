#pragma once
#include <QDialog>
#include <QTableWidget>
#include <QPushButton>
#include <QComboBox>
#include <QLineEdit>
#include<QStringList>

class CalibrationDialog : public QDialog
{
    Q_OBJECT

public:
    CalibrationDialog(QWidget* parent = nullptr,  int CalibrationIndex =1);

    ~CalibrationDialog();
    static int CalibrationIndex;
    QTableWidget* TableWidget;
    QLineEdit* NameLineEdit;
    QPushButton* OkButton;
    QPushButton* CancelButton;

    QStringList GetLeftFilePath() const;
    QStringList GetRightFilePath() const; // 以便在主窗口获取这个信息

public slots:
    void OnHeaderClicked(int section); //用于打开文件选择图片
private:
    void SetupUi();
    QComboBox* CameraComboBox;
    QStringList LeftFilePath;
    QStringList RightFilePath;//用于储存文件信息
};