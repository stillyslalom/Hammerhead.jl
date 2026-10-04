import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import jlqml

StepPage {
    title: "Results"
    guidance: "Browse the finished run: step through pairs, choose a field, and click a vector " +
              "in the viewer to inspect it."

    Label {
        text: app.resultsLabel
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }
    RowLayout {
        spacing: 8
        Button { text: "Open results…"; onClicked: openResultsDialog.open() }
        Button {
            text: "Use these settings"
            enabled: app.resultsFile !== ""
            onClicked: Julia.hh_use_result_settings()
            ToolTip.visible: hovered
            ToolTip.text: "Load the settings stored in this results file into the earlier steps"
        }
    }

    GridLayout {
        visible: app.hasResults
        columns: 2
        columnSpacing: 16
        rowSpacing: 8
        Layout.topMargin: 12

        Label { text: "Pair" }
        RowLayout {
            Slider {
                id: frameSlider
                Layout.preferredWidth: 220
                from: 1
                to: Math.max(2, app.hasResults ? app.resultFrames : 2)
                stepSize: 1
                snapMode: Slider.SnapAlways
                enabled: app.hasResults && app.resultFrames > 1
                value: app.hasResults ? app.resultFrame : 1
                onMoved: Julia.hh_result_frame(Math.round(value))
            }
            Label { text: app.hasResults ? app.resultFrame + " of " + app.resultFrames : "" }
        }

        Label { text: "Field" }
        ComboBox {
            id: fieldBox
            Layout.preferredWidth: 300
            readonly property var keys: app.hasResults ? app.resultFieldKeys.split("|") : []
            model: app.hasResults ? app.resultFieldLabels.split("|") : []
            currentIndex: Math.max(0, keys.indexOf(app.hasResults ? app.resultField : ""))
            displayText: app.hasResults ? app.resultFieldLabel : ""
            onActivated: (i) => Julia.hh_result_field(keys[i])
        }

        Label { text: "Colour range" }
        ComboBox {
            Layout.preferredWidth: 300
            textRole: "text"; valueRole: "value"
            model: [{ text: "Robust (2–98 %)", value: "robust" },
                    { text: "Full range", value: "full" }]
            currentIndex: app.hasResults && app.resultColorMode === "full" ? 1 : 0
            onActivated: Julia.hh_result_color_mode(currentValue)
        }

        Label { text: "Vectors" }
        Switch {
            checked: app.hasResults ? app.resultVectors : true
            onToggled: Julia.hh_result_vectors(checked)
        }
    }

    Frame {
        visible: app.hasResults && app.selectionText !== ""
        Layout.fillWidth: true
        Layout.topMargin: 8
        Label {
            anchors.left: parent.left
            anchors.right: parent.right
            text: app.hasResults ? app.selectionText : ""
            wrapMode: Text.WordWrap
            lineHeight: 1.2
        }
    }
    Label {
        text: app.hasResults ? app.resultsStatus : ""
        visible: text !== ""
        color: "#c42b1c"
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }

    FileDialog {
        id: openResultsDialog
        title: "Open results"
        nameFilters: ["Results (*.jld2)", "All files (*)"]
        onAccepted: Julia.hh_open_results(selectedFile.toString())
    }
}
