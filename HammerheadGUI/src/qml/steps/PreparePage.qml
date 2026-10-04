// Prepare step: sub-pages for preprocessing, mask, region of interest and
// physical scale. Each sub-page edits the settings through Julia; clicks on
// the viewer go to the open sub-page.
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import jlqml
import "prepare"

StepPage {
    title: "Prepare"
    guidance: "Get the images ready for correlation: preprocessing, a mask over regions " +
              "without valid particles, the region to analyze, and the physical scale. " +
              "Click the viewer to work on the image; drag to zoom, right-drag to pan."

    RowLayout {
        spacing: 0
        ButtonGroup { id: pageGroup }
        Repeater {
            model: [{ key: "preprocess", text: "Preprocess" }, { key: "mask", text: "Mask" },
                    { key: "roi", text: "Region" }, { key: "scale", text: "Scale" }]
            Button {
                text: modelData.text
                checkable: true
                checked: app.preparePage === modelData.key
                ButtonGroup.group: pageGroup
                onClicked: Julia.hh_set_prepare_page(modelData.key)
            }
        }
    }
    Label {
        text: app.prepareSummary
        opacity: 0.75
        wrapMode: Text.WordWrap
        Layout.fillWidth: true
    }

    PreprocessPane { visible: app.preparePage === "preprocess"; Layout.fillWidth: true }
    MaskPane { visible: app.preparePage === "mask"; Layout.fillWidth: true }
    RoiPane { visible: app.preparePage === "roi"; Layout.fillWidth: true }
    ScalePane { visible: app.preparePage === "scale"; Layout.fillWidth: true }
}
