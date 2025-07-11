#include "StudyCorr.h"
#include <QtWidgets/QApplication>

#ifdef _WIN32
#include <crtdbg.h>
#endif

int main(int argc, char *argv[])
{
#ifdef _WIN32
    _CrtSetDbgFlag(_CrtSetDbgFlag(_CRTDBG_REPORT_FLAG) | _CRTDBG_LEAK_CHECK_DF);
#endif
    QApplication a(argc, argv);
    StudyCorr w;
    w.show();
    return a.exec();
}