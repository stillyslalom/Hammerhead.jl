import QtQuick
import QtQuick.Controls
import Makie
import jlqml
ApplicationWindow {
    visible: true
    width: 760; height: 530
    MakieArea { id: surface; anchors.fill: parent; scene: plot }
    Timer { interval: 1000; running: true; repeat: false
        onTriggered: surface.grabToImage(function(result) {
            Julia.baseline_capture(result.saveToFile(capturePath)); Qt.exit(0);
        }) }
    Timer { interval: 15000; running: true; repeat: false; onTriggered: Qt.exit(3) }
}
