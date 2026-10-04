// One entry of the step rail: number, name, status dot and one-line summary.
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import jlqml

ItemDelegate {
    id: root
    property int stepNumber: 1
    property bool current: false

    height: 60
    highlighted: current
    onClicked: Julia.hh_set_step(model.key)

    function statusColor(s) {
        if (s === "ok") return "#2e9e44"
        if (s === "attention") return "#d08a00"
        if (s === "busy") return palette.highlight
        return "#a0a0a0"
    }

    contentItem: RowLayout {
        spacing: 10
        Rectangle {
            Layout.alignment: Qt.AlignVCenter
            width: 10
            height: 10
            radius: 5
            color: root.statusColor(model.status)
        }
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 1
            Label {
                text: root.stepNumber + "  " + model.label
                font.weight: root.current ? Font.DemiBold : Font.Normal
                Layout.fillWidth: true
            }
            Label {
                text: model.summary
                font.pixelSize: 11
                opacity: 0.7
                elide: Text.ElideRight
                Layout.fillWidth: true
            }
        }
    }
}
