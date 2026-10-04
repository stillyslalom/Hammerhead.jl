// Bottom bar: choose the representative pair and which frame the canvas shows.
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import jlqml

ToolBar {
    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 12
        anchors.rightMargin: 12
        spacing: 8
        enabled: app.pairCount > 0

        Label { text: "Pair" }
        ToolButton {
            text: "‹"
            enabled: app.pairIndex > 1
            onClicked: Julia.hh_select_pair(app.pairIndex - 1)
        }
        Slider {
            Layout.preferredWidth: 240
            from: 1
            to: Math.max(2, app.pairCount)
            stepSize: 1
            snapMode: Slider.SnapAlways
            value: app.pairIndex
            enabled: app.pairCount > 1
            onMoved: Julia.hh_select_pair(Math.round(value))
        }
        ToolButton {
            text: "›"
            enabled: app.pairIndex < app.pairCount
            onClicked: Julia.hh_select_pair(app.pairIndex + 1)
        }
        Label {
            text: app.pairCount > 0 ? app.pairIndex + " of " + app.pairCount : "no pairs"
            Layout.preferredWidth: 90
        }
        ToolSeparator {}
        ButtonGroup { id: frameGroup }
        Button {
            text: "Frame A"
            flat: true
            checkable: true
            checked: app.shown === "a"
            ButtonGroup.group: frameGroup
            onClicked: Julia.hh_show_frame("a")
        }
        Button {
            text: "Frame B"
            flat: true
            checkable: true
            checked: app.shown === "b"
            ButtonGroup.group: frameGroup
            onClicked: Julia.hh_show_frame("b")
        }
        Item { Layout.fillWidth: true }
        Label {
            text: app.framesSummary
            opacity: 0.75
        }
    }
}
