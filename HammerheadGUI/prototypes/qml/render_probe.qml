import QtQuick
import QtQuick.Controls
import Makie
import jlqml

ApplicationWindow {
    id: root
    // This probe is only allowed with QT_QPA_PLATFORM=offscreen.
    visible: true
    width: 760; height: 520
    title: "Hammerhead Qt bridge offscreen rendering probe"
    MakieArea { id: captureSurface; anchors.fill: parent; scene: plot }
    Timer {
        interval: 1000; running: true; repeat: false
        onTriggered: captureSurface.grabToImage(function(result) {
            Julia.record_capture(result.saveToFile(capturePath)); Qt.exit(0)
        })
    }
    Timer { interval: 15000; running: true; onTriggered: Qt.exit(2) }
}
