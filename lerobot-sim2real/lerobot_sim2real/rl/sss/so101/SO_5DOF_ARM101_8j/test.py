import xml.etree.ElementTree as ET, pprint, math, pathlib
def dump(path):
    root = ET.parse(path).getroot()
    tbl = []
    for j in root.findall('joint'):
        p = j.find('parent').attrib['link']
        c = j.find('child').attrib['link']
        o  = j.find('origin')
        xyz = o.attrib.get('xyz','0 0 0')
        rpy = o.attrib.get('rpy','0 0 0')
        tbl.append((j.attrib['name'], p, c, xyz, rpy))
    pprint.pp(tbl)
dump("so101_new_calib.urdf")
dump("so101_mod2.urdf")    


def check_parent_child(path):
    root = ET.parse(path).getroot()
    links = {l.attrib['name'] for l in root.findall('link')}
    for j in root.findall('joint'):
        p, c = j.find('parent').attrib['link'], j.find('child').attrib['link']
        if p not in links or c not in links:
            print("Broken ref:", j.attrib['name'], p, c)
check_parent_child("so101_mod2.urdf") 