import xml.etree.ElementTree as ET
import json

class S_DB:
    def __init__(self):
        self.strings = {}
        self.current_id = 0

    def add_string(self, text):
        self.strings[self.current_id] = text
        self.current_id += 1
        return self.current_id - 1

    def add_strings(self, texts):
        start_id = self.current_id
        ids = list(range(start_id, start_id + len(texts)))
        self.strings.update(dict(zip(ids, texts)))
        self.current_id += len(texts)
        return ids

    def get_string(self, string_id):
        return self.strings.get(string_id)
    
    def update_string(self, string_id, text):
        self.strings[string_id] = text

            
class HoT:
    def __init__(self):
        self.nodes: set[int] = set()
        self.hyperedges: dict[int, set] = {}
        self.node_to_edges: dict[int, set] = {}
        self.node_to_name: dict[int, str] = {}
        self.str_db = S_DB()

    def __str__(self) -> str:
        return str(self.nodes.keys())
    
    def __contains__(self, key):
        return key in self.nodes
    
    def _clean(self):
        self.nodes: set[int] = set()
        self.hyperedges: dict[int, set] = {}
        self.node_to_edges: dict[int, set] = {}
        self.str_db = S_DB()
    
    def add_node(self, text, name=None):
        text_id = self.str_db.add_string(text)
        node = text_id
        self.nodes.add(node)
        self.node_to_edges[node] = set()
        if name:
            self.node_to_name[node] = name
        return node

    def add_hyperedge(self, node_set, text):
        hyperedge_id = self.str_db.add_string(text)
        self.hyperedges[hyperedge_id] = node_set
        for n in node_set:
            self.node_to_edges[n].add(hyperedge_id)
        return hyperedge_id
    
    def remove_node(self, node):
        self.nodes.remove(node)
        for he in self.get_containing_hyperedges(node):
            self.hyperedges[he].remove(node)
    
    def remove_node_from_hyperedge(self, node, edge):
        self.hyperedges[edge].remove(node)

    def remove_hyperedge(self, edge):
        nodes = self.get_hyperedge_members(edge)
        for n in nodes:
            self.node_to_edges[n].remove(edge)
        del self.hyperedges[edge]

    def get_hyperedge_members(self, hyperedge_id):
        return self.hyperedges[hyperedge_id]

    def get_node_name(self, node_id: int):
        if node_id in self.node_to_name:
            return self.node_to_name[node_id]
        else:
            return None

    def get_text(self, text_id) -> str:
        return self.str_db.get_string(text_id)
    
    def update_text(self, text_id, new_text):
        self.str_db.update_string(text_id, new_text)

    def get_nodes(self):
        return self.nodes

    def get_hyperedges(self):
        return self.hyperedges
    
    def get_containing_hyperedges(self, node):
        return self.node_to_edges[node]

    def from_xml(self, xml_dir, fresh=True):
        tree = ET.parse(xml_dir)
        root = tree.getroot()

        if fresh:
            self._clean()

        nodes = root.find('nodes')
        for node in nodes:
            text = node.text.strip()
            if "name" in node.attrib:
                node_name = node.attrib["name"]
                self.add_node(text, name=node_name)
            else:
                self.add_node(text)                
                      

        hyperedges = root.find('hyperedges')
        for hyperedge in hyperedges:
            nodes_text = hyperedge.find("nodes").text
            nodes = set(map(int, nodes_text.split(',')))
            text = hyperedge.text.strip()
            hg_id = self.add_hyperedge(nodes, text)
            for n in nodes:
                self.node_to_edges[n].add(hg_id)

    def to_xml(self, out_file):
        root = ET.Element("HoT")
        nodes_elem = ET.SubElement(root, "nodes")
        edges_elem = ET.SubElement(root, "hyperedges")

        nodes = self.get_nodes()
        for node_id in nodes:
            node_elem = ET.SubElement(nodes_elem, "node")
            node_elem.text = self.get_text(node_id)
            if node_id in self.node_to_name:
                node_elem.set("name",self.node_to_name[node_id])

        hyperedges = self.get_hyperedges()
        if hyperedges is not None:
            for he_id in hyperedges:
                he_elem = ET.SubElement(edges_elem, "hyperedge")
                he_elem.text = self.get_text(he_id)
                he_members = self.get_hyperedge_members(he_id)

                he_members_elem = ET.SubElement(he_elem, "nodes")
                he_members_elem.text = ','.join(map(str, he_members))

        tree = ET.ElementTree(root)
        ET.indent(tree)
        ET.indent(nodes_elem,level=1)
        ET.indent(edges_elem,level=1)
        tree.write(out_file)
