// Frames by folder and file-name pattern, for one frame list (camera 0 is the
// planar window's; 1 and 2 the stereo cameras; 3 the planar per-frame mask
// images). The pattern can be typed, or inferred from two files of the list.
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import jlqml

ColumnLayout {
    id: row
    property int camera: 0
    property string noun: "frames"
    spacing: 6
    Layout.fillWidth: true

    RowLayout {
        spacing: 8
        Layout.fillWidth: true
        Label { text: "Folder" }
        TextField {
            id: dirField
            Layout.fillWidth: true
            text: app["patternDir" + row.camera]
            placeholderText: "a folder of " + row.noun
            selectByMouse: true
            onEditingFinished: if (text !== app["patternDir" + row.camera])
                Julia.hh_set_frame_pattern(row.camera, text, patternField.text)
        }
        Button { text: "Browse…"; onClicked: folderDialog.open() }
    }
    RowLayout {
        spacing: 8
        Layout.fillWidth: true
        Label { text: "Pattern" }
        TextField {
            id: patternField
            Layout.preferredWidth: 160
            text: app["pattern" + row.camera]
            selectByMouse: true
            onEditingFinished: if (text !== app["pattern" + row.camera])
                Julia.hh_set_frame_pattern(row.camera, dirField.text, text)
            ToolTip.visible: hovered
            ToolTip.text: "* matches any run of characters, ? one character"
        }
        Button {
            text: "From two files…"
            onClicked: twoFramesDialog.open()
            ToolTip.visible: hovered
            ToolTip.text: "Choose two " + row.noun + " (e.g. the first pair's); " +
                          "the folder and pattern follow from their names"
        }
        Button {
            text: "Add matching files"
            enabled: app["patternCount" + row.camera] > 0
            highlighted: enabled
            onClicked: Julia.hh_add_matching(row.camera)
        }
    }
    Label {
        text: app["patternInfo" + row.camera]
        visible: text !== ""
        opacity: 0.75
        elide: Text.ElideMiddle
        Layout.fillWidth: true
    }

    FolderDialog {
        id: folderDialog
        title: "Folder of " + row.noun
        onAccepted: Julia.hh_set_frame_pattern(row.camera, selectedFolder.toString(), patternField.text)
    }
    FileDialog {
        id: twoFramesDialog
        title: "Choose two " + row.noun
        fileMode: FileDialog.OpenFiles
        nameFilters: ["Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg)", "All files (*)"]
        onAccepted: Julia.hh_pattern_from_frames(row.camera,
                                                 selectedFiles.map(u => u.toString()).join("\n"))
    }
}
