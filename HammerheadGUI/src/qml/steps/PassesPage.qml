import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import jlqml

StepPage {
    title: "Passes"
    guidance: "Start from a preset, then adjust. Each pass refines the previous one; the first " +
              "window should be at least four times the largest displacement. The viewer " +
              "outlines each window size against the particles."

    RowLayout {
        spacing: 0
        Label { text: "Preset"; Layout.preferredWidth: 90 }
        ButtonGroup { id: presetGroup }
        Repeater {
            model: [{ key: "low", text: "Low" }, { key: "medium", text: "Medium" },
                    { key: "high", text: "High" }]
            Button {
                text: modelData.text
                checkable: true
                checked: app.preset === modelData.key
                ButtonGroup.group: presetGroup
                onClicked: Julia.hh_fill_preset(modelData.key)
            }
        }
        Label {
            text: app.preset === "custom" ? "custom" : ""
            opacity: 0.7
            Layout.leftMargin: 12
        }
    }

    // pass table
    GridLayout {
        columns: 6
        columnSpacing: 8
        rowSpacing: 6
        Layout.topMargin: 8

        Label { text: "" }
        Label { text: "Window (px)"; opacity: 0.75 }
        Label { text: "Search (px)"; opacity: 0.75 }
        Label { text: "Overlap (%)"; opacity: 0.75 }
        Label { text: "Repeats"; opacity: 0.75 }
        Label { text: "" }

        Repeater {
            model: passModel
            delegate: Label {
                Layout.row: model.number
                Layout.column: 0
                text: model.number
            }
        }
        Repeater {
            model: passModel
            delegate: SpinBox {
                Layout.row: model.number
                Layout.column: 1
                Layout.preferredWidth: 100
                from: 4; to: 2048; stepSize: 8; editable: true
                value: model.window
                onValueModified: Julia.hh_set_pass(model.number, "window", value)
            }
        }
        Repeater {
            model: passModel
            delegate: SpinBox {
                Layout.row: model.number
                Layout.column: 2
                Layout.preferredWidth: 100
                from: 4; to: 2048; stepSize: 8; editable: true
                value: model.search
                onValueModified: Julia.hh_set_pass(model.number, "search", value)
            }
        }
        Repeater {
            model: passModel
            delegate: SpinBox {
                Layout.row: model.number
                Layout.column: 3
                Layout.preferredWidth: 92
                from: 0; to: 90; stepSize: 5; editable: true
                value: Math.round(model.overlap)
                onValueModified: Julia.hh_set_pass(model.number, "overlap", value)
            }
        }
        Repeater {
            model: passModel
            delegate: SpinBox {
                Layout.row: model.number
                Layout.column: 4
                Layout.preferredWidth: 100
                from: 1; to: 10; editable: true
                enabled: app.mode !== "ensemble"
                ToolTip.visible: hovered && app.mode === "ensemble"
                ToolTip.text: "An ensemble runs each pass once; add a pass to repeat a window size"
                value: model.iterations
                onValueModified: Julia.hh_set_pass(model.number, "iterations", value)
            }
        }
        Repeater {
            model: passModel
            delegate: ToolButton {
                Layout.row: model.number
                Layout.column: 5
                text: "✕"
                enabled: passModel.rowCount() > 1
                ToolTip.visible: hovered
                ToolTip.text: "Remove pass " + model.number
                onClicked: Julia.hh_remove_pass(model.number)
            }
        }
    }

    RowLayout {
        Button { text: "Add pass"; onClicked: Julia.hh_add_pass() }
        Label { text: app.passesSummary; opacity: 0.75; Layout.leftMargin: 8 }
    }
    Label {
        text: app.passesError
        visible: text !== ""
        color: "#c42b1c"
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }

    Label { text: "Correlation"; font.weight: Font.DemiBold; Layout.topMargin: 12 }
    GridLayout {
        columns: 2
        columnSpacing: 16
        rowSpacing: 8

        Label { text: "Method" }
        ComboBox {
            Layout.preferredWidth: 220
            textRole: "text"; valueRole: "value"
            model: [{ text: "Cross-correlation", value: "cross" },
                    { text: "Phase correlation", value: "phase" }]
            currentIndex: app.correlation === "phase" ? 1 : 0
            onActivated: Julia.hh_set_option("correlation", currentValue)
        }
        Label { text: "Subpixel fit" }
        ComboBox {
            Layout.preferredWidth: 220
            textRole: "text"; valueRole: "value"
            model: [{ text: "3-point Gaussian", value: "gauss3" },
                    { text: "9-point Gaussian", value: "gauss9" },
                    { text: "2-D Gaussian fit", value: "gauss2d" }]
            currentIndex: Math.max(0, ["gauss3", "gauss9", "gauss2d"].indexOf(app.subpixel))
            onActivated: Julia.hh_set_option("subpixel", currentValue)
        }
        Label { text: "Padding and Gaussian weighting" }
        Switch {
            checked: app.accuracy
            onToggled: Julia.hh_set_option("accuracy", checked)
            ToolTip.visible: hovered
            ToolTip.text: "Most accurate (about 0.03 px RMS); slower"
        }
        Label { text: "Uncertainty on final pass" }
        Switch {
            checked: app.uncertainty
            onToggled: Julia.hh_set_option("uncertainty", checked)
            ToolTip.visible: hovered
            ToolTip.text: "Per-vector random-error estimate (Wieneke 2015); needs a converged final pass"
        }
    }

    Label { text: "Evaluation"; font.weight: Font.DemiBold; Layout.topMargin: 12 }
    GridLayout {
        columns: 2
        columnSpacing: 16
        rowSpacing: 8
        Label { text: "Mode" }
        ComboBox {
            Layout.preferredWidth: 300
            textRole: "text"; valueRole: "value"
            model: [{ text: "Per pair (time series)", value: "sequence" },
                    { text: "Ensemble (one mean field)", value: "ensemble" }]
            currentIndex: app.mode === "ensemble" ? 1 : 0
            onActivated: Julia.hh_set_mode(currentValue)
        }
        Item { width: 1; height: 1; visible: app.mode === "ensemble" }
        Label {
            text: "Sums each window's correlation over all pairs and finds one peak: for " +
                  "steady flow whose single pairs are too noisy. Each pass runs once " +
                  "(repeats are ignored; add a pass to repeat a window size)."
            visible: app.mode === "ensemble"
            opacity: 0.75
            wrapMode: Text.WordWrap
            Layout.preferredWidth: 300
        }
        Label { text: "Precision" }
        ComboBox {
            Layout.preferredWidth: 300
            textRole: "text"; valueRole: "value"
            model: [{ text: "Float64", value: "Float64" },
                    { text: "Float32 (less memory)", value: "Float32" }]
            currentIndex: app.precision === "Float32" ? 1 : 0
            onActivated: Julia.hh_set_precision(currentValue)
        }
    }
}
