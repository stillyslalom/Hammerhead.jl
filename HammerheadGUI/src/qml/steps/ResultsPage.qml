import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import jlqml

StepPage {
    title: "Results"
    guidance: "Browse the finished run: step through the results with the pair bar below " +
              "the viewer (or the arrow keys), choose a field, and inspect vectors, " +
              "profiles, and circulation with the tools below."

    Label {
        text: app.resultsLabel
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }
    RowLayout {
        spacing: 8
        Button { text: "Open results…"; onClicked: openResultsDialog.open() }
    }

    GridLayout {
        visible: app.hasResults
        columns: 2
        columnSpacing: 16
        rowSpacing: 8
        Layout.topMargin: 12

        Label { text: "Field" }
        ComboBox {
            id: fieldBox
            Layout.preferredWidth: 300
            readonly property var keys: app.hasResults ? app.resultFieldKeys.split("\n") : []
            model: app.hasResults ? app.resultFieldLabels.split("\n") : []
            currentIndex: Math.max(0, keys.indexOf(app.hasResults ? app.resultField : ""))
            displayText: app.hasResults ? app.resultFieldLabel : ""
            onActivated: (i) => Julia.hh_result_field(keys[i])
        }

        Label { text: "Color range" }
        ComboBox {
            Layout.preferredWidth: 300
            textRole: "text"; valueRole: "value"
            model: [{ text: "Percentile limits", value: "percentile" },
                    { text: "Absolute limits", value: "absolute" }]
            currentIndex: app.resultColorScale === "absolute" ? 1 : 0
            onActivated: Julia.hh_result_color_scale(currentValue)
        }
        Label { text: app.resultColorScale === "absolute" ? "Limits" : "Percentiles (%)" }
        RowLayout {
            TextField {
                id: lowField
                Layout.preferredWidth: 100
                text: app.resultColorLow
                selectByMouse: true
                onEditingFinished: if (text !== app.resultColorLow) Julia.hh_result_color_bounds(text, highField.text)
            }
            Label { text: "to" }
            TextField {
                id: highField
                Layout.preferredWidth: 100
                text: app.resultColorHigh
                selectByMouse: true
                onEditingFinished: if (text !== app.resultColorHigh) Julia.hh_result_color_bounds(lowField.text, text)
            }
        }

        Label { text: "Physical units"; visible: app.resultHasScale }
        Switch {
            visible: app.resultHasScale
            checked: app.resultPhysical
            onToggled: Julia.hh_result_physical(checked)
            ToolTip.visible: hovered
            ToolTip.text: "Off: positions in px and displacements per frame pair"
        }

        Label { text: "Vectors" }
        Switch {
            checked: app.hasResults ? app.resultVectors : true
            onToggled: Julia.hh_result_vectors(checked)
        }

        Label { text: "Flagged vectors"; visible: app.resultIsPlanar }
        ComboBox {
            visible: app.resultIsPlanar
            Layout.preferredWidth: 300
            textRole: "text"; valueRole: "value"
            model: [{ text: "Left out of derived fields", value: false },
                    { text: "Included (replaced values)", value: true }]
            currentIndex: app.resultIncludeFlagged ? 1 : 0
            onActivated: Julia.hh_result_include_flagged(currentValue)
            ToolTip.visible: hovered
            ToolTip.text: "Whether derived fields, profiles and circulation use flagged vectors " +
                          "(holding replacement values when the run replaced them)"
        }
    }

    Label {
        visible: app.hasResults && app.resultIsPlanar
        text: "Validation"
        font.weight: Font.DemiBold
        Layout.topMargin: 8
    }
    GridLayout {
        visible: app.hasResults && app.resultIsPlanar
        columns: 3
        columnSpacing: 12
        rowSpacing: 6
        Label { text: "Re-validate the results" }
        Switch {
            checked: app.resultRevalidate
            onToggled: Julia.hh_result_revalidate(checked)
            ToolTip.visible: hovered
            ToolTip.text: "Check the shown vectors again with these settings instead of the run's " +
                          "flags (display only; the results are unchanged)"
        }
        Label { text: "" }
        Label { text: "Normalized median test"; enabled: app.resultRevalidate }
        Switch {
            enabled: app.resultRevalidate
            checked: app.rvUodEnable
            onToggled: Julia.hh_result_revalidation("uod_enable", checked)
        }
        Label { text: "" }
        Label { text: "Threshold"; enabled: app.resultRevalidate && app.rvUodEnable }
        TextField {
            Layout.preferredWidth: 80
            enabled: app.resultRevalidate && app.rvUodEnable
            text: app.rvUodThreshold
            onEditingFinished: if (text !== app.rvUodThreshold) Julia.hh_result_revalidation("uod_threshold", text)
        }
        ComboBox {
            Layout.preferredWidth: 100
            enabled: app.resultRevalidate && app.rvUodEnable
            textRole: "text"; valueRole: "value"
            model: [{ text: "3 × 3", value: 1 }, { text: "5 × 5", value: 2 }, { text: "7 × 7", value: 3 }]
            currentIndex: Math.max(0, [1, 2, 3].indexOf(app.rvUodNeighborhood))
            onActivated: Julia.hh_result_revalidation("uod_neighborhood", currentValue)
        }
        Label { text: "Minimum peak ratio"; enabled: app.resultRevalidate }
        TextField {
            Layout.preferredWidth: 80
            enabled: app.resultRevalidate
            text: app.rvMinPeakRatio
            onEditingFinished: if (text !== app.rvMinPeakRatio) Julia.hh_result_revalidation("min_peak_ratio", text)
        }
        Label { text: "1 = off"; opacity: 0.75 }
        Label { text: "Replace flagged vectors"; enabled: app.resultRevalidate }
        Switch {
            enabled: app.resultRevalidate
            checked: app.rvReplace
            onToggled: Julia.hh_result_revalidation("replace", checked)
        }
        Label { text: "" }
    }

    Label {
        visible: app.hasResults
        text: "Tool"
        font.weight: Font.DemiBold
        Layout.topMargin: 8
    }
    RowLayout {
        visible: app.hasResults
        spacing: 0
        ButtonGroup { id: toolGroup }
        Repeater {
            model: [{ key: "inspect", text: "Inspect" }, { key: "profile", text: "Profile" },
                    { key: "circulation", text: "Circulation" }]
            Button {
                text: modelData.text
                checkable: true
                checked: app.resultTool === modelData.key
                enabled: modelData.key === "inspect" || app.resultToolsAvailable
                ButtonGroup.group: toolGroup
                onClicked: Julia.hh_result_tool(modelData.key)
            }
        }
        Item { implicitWidth: 12 }
        Button {
            text: "Clear"
            enabled: app.resultTool !== "inspect"
            onClicked: Julia.hh_result_clear_tool()
            ToolTip.visible: hovered
            ToolTip.text: "Remove the line or contour (Escape on the viewer)"
        }
    }
    Label {
        visible: app.hasResults
        text: app.resultTool === "profile"
              ? "Click two points on the viewer to sample the shown field along a line; " +
                "drag a point to move it. A click away from the points starts a new line."
              : app.resultTool === "circulation"
                ? "Click contour vertices on the viewer, right-click to close the contour. " +
                  "Drag a vertex to move it; click one and press Delete to remove it."
                : app.resultToolsAvailable
                  ? "Click a vector on the viewer to inspect it."
                  : "Click a vector on the viewer to inspect it. Profile and circulation " +
                    "need a planar PIV result."
        wrapMode: Text.WordWrap
        opacity: 0.75
        Layout.fillWidth: true
    }
    Frame {
        visible: app.hasResults && app.toolSummary !== ""
        Layout.fillWidth: true
        Label {
            anchors.left: parent.left
            anchors.right: parent.right
            text: app.hasResults ? app.toolSummary : ""
            wrapMode: Text.WordWrap
            lineHeight: 1.2
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
