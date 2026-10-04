// Preprocess sub-page: the ordered step list (`prepModel`), the background
// estimate, the raw/processed viewer toggle, and the correlation probe.
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import jlqml

ColumnLayout {
    spacing: 10

    Label {
        text: "Steps run in order on every frame before correlation, exactly as listed."
        wrapMode: Text.WordWrap
        opacity: 0.75
        Layout.fillWidth: true
    }

    Repeater {
        model: prepModel
        delegate: Frame {
            id: stepFrame
            Layout.fillWidth: true
            readonly property int stepNumber: model.number
            readonly property var keys: model.optionKeys === "" ? [] : model.optionKeys.split("|")
            readonly property var labels: model.optionLabels === "" ? [] : model.optionLabels.split("|")
            readonly property var values: model.optionValues === "" ? [] : model.optionValues.split("|")

            ColumnLayout {
                anchors.left: parent.left
                anchors.right: parent.right
                spacing: 6
                RowLayout {
                    Layout.fillWidth: true
                    Label {
                        text: stepFrame.stepNumber + ". " + model.label
                        font.weight: Font.DemiBold
                        Layout.fillWidth: true
                        elide: Text.ElideRight
                    }
                    ToolButton {
                        text: "↑"
                        enabled: stepFrame.stepNumber > 1
                        onClicked: Julia.hh_move_step(stepFrame.stepNumber, -1)
                        ToolTip.visible: hovered
                        ToolTip.text: "Run earlier"
                    }
                    ToolButton {
                        text: "↓"
                        enabled: stepFrame.stepNumber < prepModel.rowCount()
                        onClicked: Julia.hh_move_step(stepFrame.stepNumber, 1)
                        ToolTip.visible: hovered
                        ToolTip.text: "Run later"
                    }
                    ToolButton {
                        text: "✕"
                        onClicked: Julia.hh_remove_step(stepFrame.stepNumber)
                        ToolTip.visible: hovered
                        ToolTip.text: "Remove this step"
                    }
                }
                Flow {
                    visible: stepFrame.keys.length > 0
                    Layout.fillWidth: true
                    spacing: 12
                    Repeater {
                        model: stepFrame.keys.length
                        RowLayout {
                            spacing: 6
                            Label { text: stepFrame.labels[index] }
                            TextField {
                                Layout.preferredWidth: 80
                                text: stepFrame.values[index]
                                selectByMouse: true
                                onEditingFinished: {
                                    if (text !== stepFrame.values[index])
                                        Julia.hh_set_step_option(stepFrame.stepNumber,
                                                                 stepFrame.keys[index], text)
                                }
                            }
                        }
                    }
                }
                Label {
                    text: model.error
                    visible: text !== ""
                    color: "#c42b1c"
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
            }
        }
    }

    RowLayout {
        spacing: 8
        ComboBox {
            id: addBox
            Layout.preferredWidth: 260
            textRole: "text"
            valueRole: "value"
            model: [{ text: "Background subtraction", value: "subtract_background" },
                    { text: "Intensity cap", value: "intensity_cap" },
                    { text: "Highpass filter", value: "highpass_filter" },
                    { text: "CLAHE (local contrast)", value: "clahe" },
                    { text: "Percentile stretch", value: "percentile_stretch" },
                    { text: "Invert", value: "invert_image" },
                    { text: "Local variance normalization", value: "local_variance_normalize" }]
            currentIndex: 2
        }
        Button {
            text: "Add step"
            enabled: !app.backgroundRunning
            onClicked: {
                if (addBox.currentValue === "subtract_background")
                    Julia.hh_estimate_background(bgFrames.value)
                else
                    Julia.hh_add_step(addBox.currentValue)
            }
        }
    }
    RowLayout {
        spacing: 8
        Button {
            text: "Estimate background"
            enabled: !app.backgroundRunning && app.pairCount > 0
            onClicked: Julia.hh_estimate_background(bgFrames.value)
            ToolTip.visible: hovered
            ToolTip.text: "Pixel-wise minimum of the first frames, subtracted as the first step"
        }
        Label { text: "from the first" }
        SpinBox {
            id: bgFrames
            from: 1; to: 1000; value: 10; editable: true
            Layout.preferredWidth: 110
        }
        Label { text: "frames" }
        BusyIndicator {
            running: app.backgroundRunning
            visible: running
            implicitWidth: 28
            implicitHeight: 28
        }
    }
    Label {
        text: app.prepareStatus
        visible: text !== ""
        wrapMode: Text.WordWrap
        opacity: 0.75
        Layout.fillWidth: true
    }
    Label {
        text: "Pipeline: " + app.pipelineSummary
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }
    Label {
        text: app.previewStatus
        visible: text !== ""
        color: "#c42b1c"
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }

    Label { text: "Viewer"; font.weight: Font.DemiBold; Layout.topMargin: 8 }
    RowLayout {
        spacing: 0
        ButtonGroup { id: viewGroup }
        Button {
            text: "Raw"
            checkable: true
            checked: !app.showProcessed
            ButtonGroup.group: viewGroup
            onClicked: Julia.hh_show_processed(false)
        }
        Button {
            text: "Processed"
            checkable: true
            checked: app.showProcessed
            ButtonGroup.group: viewGroup
            onClicked: Julia.hh_show_processed(true)
        }
    }

    Label { text: "Correlation probe"; font.weight: Font.DemiBold; Layout.topMargin: 8 }
    Label {
        text: "Click the image to correlate one window of the processed pair there. " +
              "Probe a few places: a clear peak (ratio well above 1.5) means the " +
              "preprocessing suits the images. Right-click removes the probe."
        wrapMode: Text.WordWrap
        opacity: 0.75
        Layout.fillWidth: true
    }
    RowLayout {
        spacing: 8
        Label { text: "Window (px)" }
        SpinBox {
            from: 8; to: 1024; stepSize: 8; editable: true
            value: app.probeWindow
            Layout.preferredWidth: 110
            onValueModified: Julia.hh_set_probe_window(value)
        }
        Button { text: "Remove probe"; onClicked: Julia.hh_clear_probe() }
    }
    Label {
        text: app.probeSummary
        wrapMode: Text.WordWrap
        lineHeight: 1.2
        Layout.fillWidth: true
    }
}
