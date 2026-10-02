import QtQuick
import QtQuick.Controls
import Makie
import jlqml

ApplicationWindow {
    visible: false
    width: 720
    height: 480
    title: "Hidden Hammerhead QMLMakie probe"
    MakieArea {
        anchors.fill: parent
        scene: plot
        Component.onCompleted: Julia.record_probe("makie_component_created")
    }
    Timer {
        interval: 100
        repeat: false
        running: true
        onTriggered: {
            Julia.record_probe("hidden_event_loop_tick")
            Qt.exit(0)
        }
    }
}
