import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// 添加CSS样式
const style = document.createElement('style');
style.textContent = `
    .json-upload-container {
        padding: 10px;
        border: 1px solid #444;
        border-radius: 5px;
        margin: 10px 0;
        background: rgba(0,0,0,0.2);
    }
    
    .json-upload-btn {
        background: #4CAF50;
        color: white;
        padding: 8px 16px;
        border: none;
        border-radius: 3px;
        cursor: pointer;
        margin: 5px;
    }
    
    .json-upload-btn:hover {
        background: #45a049;
    }
    
    .json-file-select, .json-key-select {
        width: 100%;
        padding: 5px;
        margin: 5px 0;
        background: #222;
        color: white;
        border: 1px solid #444;
        border-radius: 3px;
    }
    
    .json-upload-status {
        margin: 5px 0;
        padding: 5px;
        font-size: 12px;
    }
    
    .json-upload-success {
        color: #4CAF50;
    }
    
    .json-upload-error {
        color: #f44336;
    }
`;
document.head.appendChild(style);

// 创建文件上传组件
function createFileUploadWidget(node, inputName) {
    const container = document.createElement('div');
    container.className = 'json-upload-container';
    
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.json';
    fileInput.style.display = 'none';
    
    const uploadBtn = document.createElement('button');
    uploadBtn.className = 'json-upload-btn';
    uploadBtn.textContent = '选择JSON文件';
    uploadBtn.onclick = () => fileInput.click();
    
    const statusDiv = document.createElement('div');
    statusDiv.className = 'json-upload-status';
    
    const fileSelect = document.createElement('select');
    fileSelect.className = 'json-file-select';
    fileSelect.innerHTML = '<option value="">选择文件...</option>';
    
    const keySelect = document.createElement('select');
    keySelect.className = 'json-key-select';
    keySelect.innerHTML = '<option value="">选择键...</option>';
    
    // 文件上传处理
    fileInput.onchange = async (e) => {
        if (e.target.files.length === 0) return;
        
        const file = e.target.files[0];
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await api.fetchApi('/api/zvnodes/json/upload_json', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok) {
                statusDiv.textContent = `上传成功: ${result.filename}`;
                statusDiv.className = 'json-upload-status json-upload-success';
                await updateFileList();
                const fileWidget = node.widgets.find(w => w.name === 'json_file');
                if (fileWidget) {
                    fileWidget.value = result.filename;
                    node.setDirtyCanvas(true);
                    if (fileWidget.callback) {
                        fileWidget.callback(fileWidget.value);
                    }
                }
                fileSelect.value = result.filename;
                await updateKeyList(result.filename);
            } else {
                statusDiv.textContent = `上传失败: ${result.error}`;
                statusDiv.className = 'json-upload-status json-upload-error';
            }
        } catch (error) {
            statusDiv.textContent = `上传错误: ${error.message}`;
            statusDiv.className = 'json-upload-status json-upload-error';
        }
        
        fileInput.value = '';
    };
    
    // 文件选择变化时更新键列表
    fileSelect.onchange = async (e) => {
        const filename = e.target.value;
        if (filename) {
            await updateKeyList(filename);
        } else {
            keySelect.innerHTML = '<option value="">选择键...</option>';
        }
        
        // 更新节点值
        const widget = node.widgets.find(w => w.name === inputName);
        if (widget) {
            widget.value = filename;
            node.setDirtyCanvas(true);
        }

        const fileWidget = node.widgets.find(w => w.name === 'json_file');
        if (fileWidget) {
            fileWidget.value = filename;
            node.setDirtyCanvas(true);
            if (fileWidget.callback) {
                fileWidget.callback(fileWidget.value);
            }
        }
    };
    
    // 键选择变化时更新节点值
    keySelect.onchange = (e) => {
        const selectedKey = e.target.value;
        // 找到对应的key选择widget并更新
        const keyWidget = node.widgets.find(w => w.name === 'selected_key');
        if (keyWidget) {
            keyWidget.value = selectedKey;
            node.setDirtyCanvas(true);
            if (keyWidget.callback) {
                keyWidget.callback(keyWidget.value);
            }
        }
    };
    
    // 更新文件列表
    async function updateFileList() {
        try {
            const response = await api.fetchApi('/api/zvnodes/json/list_json_files');
            const result = await response.json();
            
            fileSelect.innerHTML = '<option value="">选择文件...</option>';
            result.files.forEach(file => {
                const option = document.createElement('option');
                option.value = file;
                option.textContent = file;
                fileSelect.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load file list:', error);
        }
    }
    
    // 更新键列表
    async function updateKeyList(filename) {
        try {
            const response = await api.fetchApi(`/api/zvnodes/json/get_json_keys?filename=${encodeURIComponent(filename)}`);
            const result = await response.json();
            
            keySelect.innerHTML = '<option value="">选择键...</option>';
            if (result.keys) {
                result.keys.forEach(key => {
                    const option = document.createElement('option');
                    option.value = key;
                    option.textContent = key;
                    keySelect.appendChild(option);
                });
                // 更新节点值
                const widget = node.widgets.find(w => w.name === 'selected_key');
                if (widget) {
                    widget.options.values = result.keys;
                    widget.value = result.keys[0];
                    node.setDirtyCanvas(true);
                    if (widget.callback) {
                        widget.callback(widget.value);
                    }
                }
            }
        } catch (error) {
            console.error('Failed to load keys:', error);
            statusDiv.textContent = `加载键失败: ${error.message}`;
            statusDiv.className = 'json-upload-status json-upload-error';
        }
    }
    
    // 初始化时加载文件列表
    setTimeout(() => {
        updateFileList();
    }, 100);
    
    container.appendChild(uploadBtn);
    container.appendChild(fileInput);
    container.appendChild(fileSelect);
    container.appendChild(keySelect);
    container.appendChild(statusDiv);
    
    return container;
}

// 扩展节点
app.registerExtension({
    name: "zvnodes.jsonreader",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "JsonReaderZV") {
            // 重写INPUT_TYPES以支持动态选项
            const originalInputTypes = nodeType.INPUT_TYPES;
            nodeType.INPUT_TYPES = function() {
                const inputs = originalInputTypes ? originalInputTypes.call(this) : {
                    "required": {
                        "json_file": ("STRING", {"default": ""}),
                        "selected_key": ("STRING",{"default": ""}),
                    }
                };
                return inputs;
            };
        }
    },
    async nodeCreated(node, app) {
        if (node.comfyClass === "JsonReaderZV") {
            // 节点创建时的处理
            const nodeId = node.id;
            
            // 添加上传控件
            const uploadContainer = createFileUploadWidget(node, "json_file");
            node.addDOMWidget("json_upload", "upload", uploadContainer, {
                serialize: false,
                hideOnZoom: false
            });
        }
    },
    
    async setup() {
        // 监听节点验证事件
        const originalValidateLinks = app.graph.validateLinks;
        app.graph.validateLinks = function() {
            // 在验证前更新所有JSON Dropdown节点的选项
            app.graph._nodes.forEach(node => {
                if (node.comfyClass === "JsonReaderZV") {
                    const keyWidget = node.widgets.find(w => w.name === 'selected_key');
                    if (keyWidget && nodeOptionsMap.has(node.id)) {
                        const options = nodeOptionsMap.get(node.id);
                        if (keyWidget.options.toString() !== options.toString()) {
                            keyWidget.options = options;
                        }
                    }
                }
            });
            return originalValidateLinks ? originalValidateLinks.call(this) : true;
        };
    }
});

