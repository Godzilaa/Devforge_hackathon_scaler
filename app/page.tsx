'use client';

import dynamic from 'next/dynamic';
import { useState, useEffect } from 'react';
import { 
  Play, 
  Save, 
  Folder, 
  File, 
  Terminal, 
  Settings, 
  ChevronDown, 
  ChevronRight,
  Plus,
  X,
  FilePlus,
  FolderPlus
} from 'lucide-react';

const MonacoEditor = dynamic(() => import('@monaco-editor/react'), {
  ssr: false,
  loading: () => (
    <div className="h-full w-full flex items-center justify-center bg-[#1e1e1e] text-white">
      Loading Editor...
    </div>
  ),
});

interface FileItem {
  id: string;
  name: string;
  type: 'file' | 'folder';
  content?: string;
  language?: string;
  children?: FileItem[];
  isOpen?: boolean;
}

interface Tab {
  id: string;
  name: string;
  content: string;
  language: string;
  isDirty: boolean;
}

const defaultFiles: FileItem[] = [
  {
    id: 'src',
    name: 'src',
    type: 'folder',
    isOpen: true,
    children: [
      {
        id: 'main.py',
        name: 'main.py',
        type: 'file',
        language: 'python',
        content: `# Welcome to CodeIDE! 🚀
print("Hello, World!")

def calculate(a, b, operation):
    if operation == '+':
        return a + b
    elif operation == '-':
        return a - b
    elif operation == '*':
        return a * b
    elif operation == '/':
        return a / b if b != 0 else "Error"
    return "Invalid"

result = calculate(10, 5, '+')
print(f"10 + 5 = {result}")

numbers = [1, 2, 3, 4, 5]
print(f"Sum: {sum(numbers)}")
print(f"Average: {sum(numbers) / len(numbers)}")`
      },
      {
        id: 'app.js',
        name: 'app.js',
        type: 'file',
        language: 'javascript',
        content: `// JavaScript example
console.log("Welcome to CodeIDE!");

const greeting = (name = "World") => {
    return \`Hello, \${name}!\`;
};

console.log(greeting());
console.log(greeting("CodeIDE"));

const numbers = [1, 2, 3, 4, 5];
const sum = numbers.reduce((acc, n) => acc + n, 0);
console.log("Sum:", sum);`
      }
    ]
  },
  {
    id: 'README.md',
    name: 'README.md',
    type: 'file',
    language: 'markdown',
    content: `# CodeIDE - Web-based Development Environment

🚀 **Welcome to CodeIDE!**

## Features

- 📝 Monaco Editor (VS Code engine)
- 📁 File Explorer with create/delete
- 🎨 Syntax Highlighting
- 💾 Auto-save
- 🌙 Dark Theme
- ⚡ Fast Performance

## Getting Started

1. Use the file explorer to navigate files
2. Click (+) buttons to add new files/folders
3. Start coding!
4. Use the run button to execute code

**Happy Coding!** 🎉`
  }
];

export default function IDEPage() {
  const [files, setFiles] = useState<FileItem[]>(defaultFiles);
  const [tabs, setTabs] = useState<Tab[]>([]);
  const [activeTab, setActiveTab] = useState<string | null>(null);
  const [terminalOutput, setTerminalOutput] = useState<string[]>(["Welcome to CodeIDE Terminal! 🚀\nType 'help' for available commands."]);
  const [terminalInput, setTerminalInput] = useState('');
  const [showTerminal, setShowTerminal] = useState(false);
  const [sidebarWidth] = useState(300);
  const [showNewFileDialog, setShowNewFileDialog] = useState(false);
  const [showNewFolderDialog, setShowNewFolderDialog] = useState(false);
  const [newItemName, setNewItemName] = useState('');
  
  useEffect(() => {
    const mainFile = findFileById(files, 'main.py');
    if (mainFile) {
      openFile(mainFile);
    }
  }, []);
  
  const findFileById = (items: FileItem[], id: string): FileItem | null => {
    for (const item of items) {
      if (item.id === id) return item;
      if (item.children) {
        const found = findFileById(item.children, id);
        if (found) return found;
      }
    }
    return null;
  };
  
  const getLanguageFromExtension = (filename: string): string => {
    const ext = filename.split('.').pop()?.toLowerCase();
    const langMap: { [key: string]: string } = {
      'py': 'python',
      'js': 'javascript',
      'ts': 'typescript',
      'jsx': 'javascript',
      'tsx': 'typescript',
      'html': 'html',
      'css': 'css',
      'json': 'json',
      'md': 'markdown',
      'txt': 'plaintext'
    };
    return langMap[ext || ''] || 'plaintext';
  };
  
  const createNewFile = () => {
    if (!newItemName.trim()) return;
    
    const newFile: FileItem = {
      id: `file_${Date.now()}`,
      name: newItemName,
      type: 'file',
      language: getLanguageFromExtension(newItemName),
      content: ''
    };
    
    setFiles(prev => [...prev, newFile]);
    addToTerminal(`✅ Created new file: ${newItemName}`);
    setShowNewFileDialog(false);
    setNewItemName('');
    openFile(newFile);
  };
  
  const createNewFolder = () => {
    if (!newItemName.trim()) return;
    
    const newFolder: FileItem = {
      id: `folder_${Date.now()}`,
      name: newItemName,
      type: 'folder',
      isOpen: true,
      children: []
    };
    
    setFiles(prev => [...prev, newFolder]);
    addToTerminal(`✅ Created new folder: ${newItemName}`);
    setShowNewFolderDialog(false);
    setNewItemName('');
  };
  
  const toggleFolder = (folderId: string) => {
    const toggleFolderRecursive = (items: FileItem[]): FileItem[] => {
      return items.map(item => {
        if (item.id === folderId && item.type === 'folder') {
          return { ...item, isOpen: !item.isOpen };
        }
        if (item.children) {
          return { ...item, children: toggleFolderRecursive(item.children) };
        }
        return item;
      });
    };
    
    setFiles(prev => toggleFolderRecursive(prev));
  };
  
  const openFile = (file: FileItem) => {
    if (file.type !== 'file') return;
    
    const existingTab = tabs.find(tab => tab.id === file.id);
    if (existingTab) {
      setActiveTab(file.id);
      return;
    }
    
    const newTab: Tab = {
      id: file.id,
      name: file.name,
      content: file.content || '',
      language: file.language || 'plaintext',
      isDirty: false
    };
    
    setTabs(prev => [...prev, newTab]);
    setActiveTab(file.id);
  };
  
  const closeTab = (tabId: string, e?: React.MouseEvent) => {
    e?.stopPropagation();
    setTabs(prev => prev.filter(tab => tab.id !== tabId));
    if (activeTab === tabId) {
      const remainingTabs = tabs.filter(tab => tab.id !== tabId);
      setActiveTab(remainingTabs.length > 0 ? remainingTabs[remainingTabs.length - 1].id : null);
    }
  };
  
  const updateTabContent = (tabId: string, content: string) => {
    setTabs(prev => prev.map(tab => 
      tab.id === tabId 
        ? { ...tab, content, isDirty: true }
        : tab
    ));
  };
  
  const saveFile = () => {
    if (!activeTab) return;
    setTabs(prev => prev.map(tab => 
      tab.id === activeTab 
        ? { ...tab, isDirty: false }
        : tab
    ));
    addToTerminal(`✅ File '${tabs.find(t => t.id === activeTab)?.name}' saved successfully!`);
  };
  
  const runCode = () => {
    const activeTabData = tabs.find(tab => tab.id === activeTab);
    if (!activeTabData) return;
    
    addToTerminal(`\n🚀 Running ${activeTabData.name}...`);
    
    if (activeTabData.language === 'python') {
      addToTerminal("Python execution simulation:");
      addToTerminal("Hello, World!");
      addToTerminal("10 + 5 = 15");
      addToTerminal("Sum: 15");
      addToTerminal("Average: 3.0");
    } else if (activeTabData.language === 'javascript') {
      addToTerminal("JavaScript execution simulation:");
      addToTerminal("Welcome to CodeIDE!");
      addToTerminal("Hello, World!");
      addToTerminal("Hello, CodeIDE!");
      addToTerminal("Sum: 15");
    } else {
      addToTerminal(`File type: ${activeTabData.language}`);
      addToTerminal("(Execution simulation)");
    }
    
    addToTerminal("✅ Execution completed.\n");
  };
  
  const addToTerminal = (text: string) => {
    setTerminalOutput(prev => [...prev, text]);
  };
  
  const handleTerminalCommand = (command: string) => {
    addToTerminal(`$ ${command}`);
    
    switch (command.toLowerCase()) {
      case 'help':
        addToTerminal("Available commands:");
        addToTerminal("  help     - Show this help message");
        addToTerminal("  clear    - Clear terminal output");
        addToTerminal("  ls       - List files");
        addToTerminal("  pwd      - Print working directory");
        addToTerminal("  date     - Show current date");
        break;
      case 'clear':
        setTerminalOutput(["Welcome to CodeIDE Terminal! 🚀\nType 'help' for available commands."]);
        break;
      case 'ls':
        files.forEach(f => addToTerminal(f.type === 'folder' ? `${f.name}/` : f.name));
        break;
      case 'pwd':
        addToTerminal("/workspace");
        break;
      case 'date':
        addToTerminal(new Date().toString());
        break;
      default:
        addToTerminal(`Command not found: ${command}`);
        addToTerminal("Type 'help' for available commands.");
    }
    
    setTerminalInput('');
  };
  
  const renderFileTree = (items: FileItem[], depth = 0) => {
    return items.map(item => (
      <div key={item.id} className="select-none">
        <div 
          className="flex items-center py-1 px-2 hover:bg-gray-700 cursor-pointer group"
          onClick={() => item.type === 'file' ? openFile(item) : toggleFolder(item.id)}
          style={{ paddingLeft: `${depth * 16 + 8}px` }}
        >
          {item.type === 'folder' && (
            item.isOpen ? <ChevronDown className="w-4 h-4 mr-1" /> : <ChevronRight className="w-4 h-4 mr-1" />
          )}
          
          {item.type === 'folder' ? (
            <Folder className="w-4 h-4 mr-2 text-blue-400" />
          ) : (
            <File className="w-4 h-4 mr-2 text-gray-400" />
          )}
          
          <span className="text-sm text-gray-300 truncate">
            {item.name}
          </span>
        </div>
        
        {item.type === 'folder' && item.isOpen && item.children && (
          <div>
            {renderFileTree(item.children, depth + 1)}
          </div>
        )}
      </div>
    ));
  };
  
  const activeTabData = tabs.find(tab => tab.id === activeTab);

  return (
    <div className="h-screen flex flex-col bg-[#1e1e1e] text-white">
      {/* New File Dialog */}
      {showNewFileDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-[#2d2d30] p-6 rounded-lg shadow-xl w-96">
            <h3 className="text-lg font-semibold mb-4">Create New File</h3>
            <input
              type="text"
              value={newItemName}
              onChange={(e) => setNewItemName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && createNewFile()}
              placeholder="Enter file name (e.g., script.py)"
              className="w-full bg-[#1e1e1e] text-white px-3 py-2 rounded border border-gray-600 focus:border-blue-500 outline-none"
              autoFocus
            />
            <div className="flex justify-end space-x-2 mt-4">
              <button
                onClick={() => { setShowNewFileDialog(false); setNewItemName(''); }}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded"
              >
                Cancel
              </button>
              <button
                onClick={createNewFile}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}

      {/* New Folder Dialog */}
      {showNewFolderDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-[#2d2d30] p-6 rounded-lg shadow-xl w-96">
            <h3 className="text-lg font-semibold mb-4">Create New Folder</h3>
            <input
              type="text"
              value={newItemName}
              onChange={(e) => setNewItemName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && createNewFolder()}
              placeholder="Enter folder name"
              className="w-full bg-[#1e1e1e] text-white px-3 py-2 rounded border border-gray-600 focus:border-blue-500 outline-none"
              autoFocus
            />
            <div className="flex justify-end space-x-2 mt-4">
              <button
                onClick={() => { setShowNewFolderDialog(false); setNewItemName(''); }}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded"
              >
                Cancel
              </button>
              <button
                onClick={createNewFolder}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="h-12 bg-[#2d2d30] flex items-center justify-between px-4 border-b border-gray-700">
        <div className="flex items-center space-x-4">
          <h1 className="text-lg font-semibold text-white">CodeIDE</h1>
          <div className="flex space-x-2">
            <button
              onClick={saveFile}
              disabled={!activeTabData?.isDirty}
              className={`p-2 rounded transition-colors ${
                activeTabData?.isDirty 
                  ? 'bg-blue-600 hover:bg-blue-700' 
                  : 'bg-gray-600 cursor-not-allowed'
              }`}
              title="Save (Ctrl+S)"
            >
              <Save className="w-4 h-4" />
            </button>
            
            <button
              onClick={runCode}
              disabled={!activeTab}
              className={`p-2 rounded transition-colors ${
                activeTab 
                  ? 'bg-green-600 hover:bg-green-700' 
                  : 'bg-gray-600 cursor-not-allowed'
              }`}
              title="Run Code"
            >
              <Play className="w-4 h-4" />
            </button>
            
            <button
              onClick={() => setShowTerminal(!showTerminal)}
              className={`p-2 rounded transition-colors ${
                showTerminal 
                  ? 'bg-yellow-600 hover:bg-yellow-700' 
                  : 'bg-gray-600 hover:bg-gray-700'
              }`}
              title="Toggle Terminal"
            >
              <Terminal className="w-4 h-4" />
            </button>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <Settings className="w-5 h-5 text-gray-400 hover:text-white cursor-pointer" />
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <div 
          className="bg-[#252526] border-r border-gray-700 flex flex-col"
          style={{ width: sidebarWidth }}
        >
          <div className="p-3 border-b border-gray-700 flex items-center justify-between">
            <h2 className="text-sm font-medium text-gray-300 uppercase tracking-wide">
              Explorer
            </h2>
            <div className="flex space-x-1">
              <button
                onClick={() => setShowNewFileDialog(true)}
                className="p-1 hover:bg-gray-600 rounded"
                title="New File"
              >
                <FilePlus className="w-4 h-4" />
              </button>
              <button
                onClick={() => setShowNewFolderDialog(true)}
                className="p-1 hover:bg-gray-600 rounded"
                title="New Folder"
              >
                <FolderPlus className="w-4 h-4" />
              </button>
            </div>
          </div>
          
          <div className="flex-1 overflow-auto">
            {renderFileTree(files)}
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          {/* Tabs */}
          {tabs.length > 0 && (
            <div className="flex bg-[#2d2d30] border-b border-gray-700 overflow-x-auto">
              {tabs.map(tab => (
                <div
                  key={tab.id}
                  className={`flex items-center px-3 py-2 border-r border-gray-700 cursor-pointer group min-w-0 ${
                    activeTab === tab.id 
                      ? 'bg-[#1e1e1e] text-white' 
                      : 'bg-[#2d2d30] text-gray-400 hover:text-white'
                  }`}
                  onClick={() => setActiveTab(tab.id)}
                >
                  <File className="w-4 h-4 mr-2 flex-shrink-0" />
                  <span className="text-sm truncate mr-2">
                    {tab.name}
                    {tab.isDirty && <span className="ml-1 text-orange-400">•</span>}
                  </span>
                  <button
                    onClick={(e) => closeTab(tab.id, e)}
                    className="opacity-0 group-hover:opacity-100 hover:bg-gray-600 p-1 rounded transition-opacity"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Editor */}
          <div className={`flex-1 ${showTerminal ? 'h-1/2' : 'h-full'}`}>
            {activeTabData ? (
              <MonacoEditor
                height="100%"
                language={activeTabData.language}
                value={activeTabData.content}
                onChange={(value) => updateTabContent(activeTab!, value || '')}
                theme="vs-dark"
                options={{
                  fontSize: 14,
                  minimap: { enabled: false },
                  scrollBeyondLastLine: false,
                  wordWrap: 'on',
                  automaticLayout: true,
                }}
              />
            ) : (
              <div className="h-full flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <File className="w-16 h-16 mx-auto mb-4 opacity-50" />
                  <p className="text-lg mb-2">Welcome to CodeIDE!</p>
                  <p className="text-sm">Open a file from the explorer to start editing</p>
                </div>
              </div>
            )}
          </div>

          {/* Terminal */}
          {showTerminal && (
            <div className="h-1/2 bg-[#1e1e1e] border-t border-gray-700 flex flex-col">
              <div className="p-2 bg-[#2d2d30] border-b border-gray-700 flex items-center">
                <Terminal className="w-4 h-4 mr-2" />
                <span className="text-sm text-gray-300">Terminal</span>
                <button
                  onClick={() => setShowTerminal(false)}
                  className="ml-auto p-1 hover:bg-gray-600 rounded"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              
              <div className="flex-1 p-3 overflow-auto font-mono text-sm">
                {terminalOutput.map((line, index) => (
                  <div key={index} className="mb-1 text-gray-300">
                    {line}
                  </div>
                ))}
                
                <div className="flex items-center mt-2">
                  <span className="text-green-400 mr-2">$</span>
                  <input
                    type="text"
                    value={terminalInput}
                    onChange={(e) => setTerminalInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        handleTerminalCommand(terminalInput);
                      }
                    }}
                    className="flex-1 bg-transparent outline-none text-white"
                    placeholder="Type a command..."
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
