#ifndef _DARTLIB_MESH_3D_H_
#define _DARTLIB_MESH_3D_H_

#include <time.h>
#include <unordered_map>
#include <queue>
#include <string>
#include <vector>
#include <list>
#include <fstream>

#include "Geometry/Point.h"
#include "Parser/strutil.h"
#include "Utils/IO.h"
#ifdef USING_MEMORY_POOL
#include "Utils/MemoryPool.h"
#endif
#include "Iterators_3.h"
#include "Boundary_3.h"

#include "MapKeys.h"

#define MAX_LINE 1024

#define T_TYPENAME template <typename tVertex, typename tEdge, typename tFace, typename tVolume, typename tDart>
#define TBASEMESH TBaseMesh_3<tVertex, tEdge, tFace, tVolume, tDart>

namespace DartLib
{
/*! \class TBaseMesh_3 BaseMesh_3.h "BaseMesh_3.h"
 *  \brief TBaseMesh_3, base class for all types of 3d-mesh classes
 *
 *  \tparam tVertex vertex class, derived from DartLib::CVertex_3 class
 *  \tparam tEdge   edge   class, derived from DartLib::CEdge_3   class
 *  \tparam tFace   face   class, derived from DartLib::CFace_3   class
 *  \tparam tVolume volume class, derived from DartLib::CVolume_3 class
 *  \tparam tDart   dart   class, derived from DartLib::CDart_3   class
 */
T_TYPENAME
class TBaseMesh_3
{
  public:
    using CVertex              = tVertex;
    using CEdge                = tEdge;
    using CFace                = tFace;
    using CVolume              = tVolume;
    using CDart                = tDart;

    using CBoundary            = TBoundary_3<TBASEMESH>;

    using VertexIterator       = Dim3::VertexIterator<TBASEMESH>;
    using EdgeIterator         = Dim3::EdgeIterator  <TBASEMESH>;
    using FaceIterator         = Dim3::FaceIterator  <TBASEMESH>;
    using VolumeIterator       = Dim3::VolumeIterator<TBASEMESH>;
    using DartIterator         = Dim3::DartIterator  <TBASEMESH>;

    using VertexVertexIterator = Dim3::VertexVertexIterator<TBASEMESH>;
    using VertexEdgeIterator   = Dim3::VertexEdgeIterator  <TBASEMESH>;
    using VertexFaceIterator   = Dim3::VertexFaceIterator  <TBASEMESH>;
    using VertexVolumeIterator = Dim3::VertexVolumeIterator<TBASEMESH>;

    using EdgeVolumeIterator   = Dim3::EdgeVolumeIterator  <TBASEMESH>;
    using EdgeFaceIterator     = Dim3::EdgeFaceIterator    <TBASEMESH>;

    using FaceDartIterator     = Dim3::FaceDartIterator    <TBASEMESH>;
    using FaceVertexIterator   = Dim3::FaceVertexIterator  <TBASEMESH>;
    using FaceEdgeIterator     = Dim3::FaceEdgeIterator    <TBASEMESH>;

    using VolumeVertexIterator = Dim3::VolumeVertexIterator<TBASEMESH>;
    using VolumeEdgeIterator   = Dim3::VolumeEdgeIterator  <TBASEMESH>;
    using VolumeFaceIterator   = Dim3::VolumeFaceIterator  <TBASEMESH>;
    
    /*!
     *  Constructor function
     */
    TBaseMesh_3(){};

    /*!
     *  Destructor function
     */
    ~TBaseMesh_3() { unload(); };


    /*=============================================================
                   Build the dart data structure
    =============================================================*/

    /*!
     *  Unload data
     */
    void unload();

    /*!
     *  Load mesh from the input file
     *  \param input: the input name of the mesh file
     */
    void read(const std::string& input);

    /*!
     *  Write mesh to the output file
     *  \param output: the output name of the mesh file
     */
    void write(const std::string& output);

    /*!
     *  Load mesh from point cloud and volumes
     *  \param vertices: the input point cloud
     *  \param  volumes: the indices of points to construct volumes
     */
    void load(const std::vector<CPoint>            & vertices, 
              const std::vector< std::vector<int> >& volumes);

    /*!
     *  Load mesh from point cloud and volumes
     *  \param  vert_id_point: a map with key(vid) and value(point)
     *  \param volume_id_vids: a map with key(fid) and value(the vertex ids to construct volume)
     */
    void load(const std::map<int, CPoint>          & vert_id_point, 
              const std::map<int, std::vector<int>>& volume_id_vids);

    /*!
     *  Load attributes for darts and cells
     *  \param   vert_id_idx: vertex id -> vertex index
     *  \param   vert_id_str: vertex id -> vertex string
     *  \param volume_id_idx: volume id -> face index
     *  \param volume_id_str: volume id -> face string
     *  \param    edge_attrs: (vid1, vid2) -> string
     */
    void load_attributes(
        const std::map<int, std::string>& vert_id_str,
        const std::map<int, std::string>& volume_id_str,
        const std::vector<std::tuple<int, int, std::string>>& edge_attrs);

    /*!
     *  Add a vertex cell.
     *  The vertex dart will be set in the processing of 
     *  creating 1-cell, so it returns a vertex pointer here.
     *  \param vid: vertex id
     *  \return the added vertex pointer
     */
    CVertex* create_vertex(int vid);

    /*!
     *  Add a volume cell
     *  \param vert_ids: ids of the vertices to construct a volume
     *  \param    volid: volume id
     *  \return a dart pointer of the volume
     */
    CDart* create_volume(const std::vector<int>& vert_ids, int volid);

    /*!
     *  Add a face cell
     *  \param pVol: volume pointer the face attaches to.
     *  \param indices: indices of the vertices to construct a face
     *  \return a dart pointer of the face
     */
    CDart* create_face(CVolume* pVol, const std::vector<int>& indices);

    /*!
     *  Add a edge cell
     *  \param pVol: volume pointer the edge attaches to.
     *  \param pF: face pointer the edge attaches to.
     *  \param indices: indices of the vertices to construct an edge
     *  \return a dart pointer of the edge
     */
    CDart* create_edge(CVolume* pVol, CFace* pF, const std::vector<int>& indices);

    /*!
     *  Link the two faces
     *  \param f1: index of the first face
     *  \param f2: index of the second face     *
     */
    void link_faces(CDart* f1, CDart* f2);

    /*!
     *  Link the two edges
     *  \param e1: index of the first edge
     *  \param e2: index of the second edge
     *
     */
    void link_edges(CDart* e1, CDart* e2);

    /*!
     *  Link the two volumes
     *  \param vol1: index of the first volume
     *  \param vol2: index of the second volume
     *
     */
    void link_volumes(CDart* vol1, CDart* vol2);


    /*=============================================================
                 Access dart and i-cell from index or id

        Recommend to use iterators instead of the following usage.
            for(int i = 0; i < darts().size(); ++i)
            {
                CDart * pD = dart(i);
                ...
            }
    =============================================================*/

    /*
    CDart*   dart  (int index) { return m_darts[index];    };
    CVertex* vertex(int index) { return m_vertices[index]; };
    CEdge*   edge  (int index) { return m_edges[index];    };
    CFace*   face  (int index) { return m_faces[index];    };
    CVolume* volume(int index) { return m_volumes[index];  };
   */
    CVertex* id_vertex(int id) { return m_map_vertex.find(id) == m_map_vertex.end() ? NULL : m_map_vertex[id]; };
    CVolume* id_volume(int id) { return m_map_volume.find(id) == m_map_volume.end() ? NULL : m_map_volume[id]; };


    /*=============================================================
                   Access container of dart and i-cell

        Recommend to use iterators instead of the container.
    =============================================================*/

    std::list<CDart  *>& darts()    { return m_darts;    };
    std::list<CVertex*>& vertices() { return m_vertices; };
    std::list<CEdge*>  & edges()    { return m_edges;    };
    std::list<CFace  *>& faces()    { return m_faces;    };
    std::list<CVolume*>& volumes()  { return m_volumes;  };

    std::unordered_map<int, CVertex*>& map_vertex(){ return m_map_vertex;};
    std::unordered_map<int, CVolume*>& map_volume(){ return m_map_volume;};
    

    /*=============================================================
                Access dart from i-cell and vice versa
                              (Part I)
    =============================================================*/
    
    CVertex*C0(CDart* dart)  { return dart != NULL ? (CVertex*)dart->cell(0) : NULL;};
    CEdge*  C1(CDart* dart)  { return dart != NULL ? (CEdge*)  dart->cell(1) : NULL;};
    CFace*  C2(CDart* dart)  { return dart != NULL ? (CFace*)  dart->cell(2) : NULL;};
    CVolume*C3(CDart* dart)  { return dart != NULL ? (CVolume*)dart->cell(3) : NULL;};
    CDart*  D (CVertex* vert){ return vert != NULL ? (CDart*)  vert->dart()  : NULL;};
    CDart*  D (CEdge*   edge){ return edge != NULL ? (CDart*)  edge->dart()  : NULL;};
    CDart*  D (CFace*   face){ return face != NULL ? (CDart*)  face->dart()  : NULL;};
    CDart*  D (CVolume*  vol){ return  vol != NULL ? (CDart*)   vol->dart()  : NULL;};

    /*!
     *  The beta function in combinatorial map
     *  \param i: represents for function beta_i, i \in {1, 2, 3}
     *  \param dart: the input dart
     */
    CDart* beta(int i, CDart* dart) { return (CDart*) dart->beta(i); };


    /*=============================================================
                Access dart from i-cell and vice versa
                              (Part II)
    =============================================================*/

    CVertex* dart_vertex(CDart* dart) { return C0(dart); };
    CVertex* dart_target(CDart* dart) { return C0(dart); };
    CVertex* dart_source(CDart* dart);
    CEdge  * dart_edge  (CDart* dart) { return C1(dart); };
    CFace  * dart_face  (CDart* dart) { return C2(dart); };
    CVolume* dart_volume(CDart* dart) { return C3(dart); };
    CDart  * dart_prev  (CDart* dart);
    CDart  * dart_next  (CDart* dart) { return beta(1, dart); };
    CDart  * dart_sym   (CDart* dart) { return beta(2, dart); };
    //CDart  * dart       (CVertex* vert, CFace* face);
    //CDart  * dart       (CVertex* vert, CVolume* volume);

    CEdge  * vertex_edge(CVertex* v0, CVertex* v1);
    CDart  * vertex_dart(CVertex* vert) { return D(vert); };
    //CDart  * vertex_dart(CVertex* source, CVertex* target);
    
    CVertex* edge_vertex(CEdge* edge, int index);
    
    CDart  * face_dart  (CFace* face)  { return D(face);};

    CDart  * volume_dart(CVolume* vol) { return D(vol); };


    /*=============================================================
                Access neighbouring darts from i-cell
                              (Part III)
    =============================================================*/

    /*!
     *  Get all darts incident to the given vertex.
     *  \param vertex: current vertex
     *  \return all incident darts
     */
    std::vector<CDart*> vertex_incident_darts(CVertex* vertex);

    /*!
     *  Get all darts incident to the given vertex on volume.
     *  \param vertex: current vertex
     *  \param volume: the volume where the darts attached on
     *  \return all incident darts attached on the volume
     */
    std::vector<CDart*> vertex_incident_darts(CVertex* vertex, CVolume* volume);

    /*!
     *  Get all 2-cells attached on a 3-cell.
     *  \param volume: the input volume
     *  \return the darts represent for the 2-cells.
     */
    std::vector<CDart*> C32(CVolume* volume);

    /*!
     *  Get all 1-cells attached on a 3-cell.
     *  \param volume: the input volume
     *  \return the darts represent for the 1-cells.
     */
    //std::vector<CDart*> C31(CVolume* volume);


    /*=============================================================
                         Boundary detectors
    =============================================================*/

    bool boundary(CDart* dart) { return    dart->boundary(); };
    bool boundary(CVertex*  v);
    bool boundary(CEdge* edge) { return D(edge)->boundary(); };
    bool boundary(CFace* face) { return D(face)->boundary(); };

    bool consistent(CDart* dart, CVolume* vol) { return D(vol) == dart && C3(dart) == vol; };
    
    /*=============================================================
                    Number of all kinds of element
    =============================================================*/

    size_t num_vertices() { return m_vertices.size(); };
    size_t num_edges()    { return m_edges.size();    };
    size_t num_faces()    { return m_faces.size();    };
    size_t num_volumes()  { return m_volumes.size();  };
    size_t num_darts()    { return m_darts.size();    };

  protected:
    void _post_processing();

  protected:

#ifdef USING_MEMORY_POOL
    MemoryPool<CDart  , sizeof(CDart)  * 4096 * 16> m_dart_pool;
    MemoryPool<CVertex, sizeof(CVertex)* 4096>      m_vertex_pool;
    MemoryPool<CEdge,   sizeof(CEdge)  * 4096>      m_edge_pool;
    MemoryPool<CFace,   sizeof(CFace)  * 4096 * 2 > m_face_pool;
    MemoryPool<CVolume, sizeof(CVolume)* 4096>      m_volume_pool;
#endif

    std::list<CDart  *> m_darts;
    std::list<CVertex*> m_vertices;
    std::list<CEdge  *> m_edges;
    std::list<CFace  *> m_faces;
    std::list<CVolume*> m_volumes;

    std::unordered_map<int, CVertex*> m_map_vertex;
    std::unordered_map<int, CVolume*> m_map_volume;

    std::unordered_map<EdgeMapKey, CEdge*, EdgeMapKey_hasher> m_map_edge_keys;
    std::unordered_map<FaceMapKey, CFace*, FaceMapKey_hasher> m_map_face_keys;
};

T_TYPENAME
void TBASEMESH::unload()
{
#ifdef USING_MEMORY_POOL
    for (auto v : m_darts)    m_dart_pool.deleteElement(v);
    for (auto v : m_vertices) m_vertex_pool.deleteElement(v);
    for (auto v : m_edges)    m_edge_pool.deleteElement(v);
    for (auto v : m_faces)    m_face_pool.deleteElement(v);
    for (auto v : m_volumes)  m_volume_pool.deleteElement(v);
#else
    for (auto v : m_darts)    delete v;
    for (auto v : m_vertices) delete v;
    for (auto v : m_edges)    delete v;
    for (auto v : m_faces)    delete v;
    for (auto v : m_volumes)  delete v;
#endif
    
    m_darts.clear();
    m_vertices.clear();
    m_edges.clear();
    m_faces.clear();
    m_volumes.clear();

    m_map_vertex.clear();
    m_map_volume.clear();

    m_map_edge_keys.clear();
    m_map_face_keys.clear();
}

T_TYPENAME
void TBASEMESH::read(const std::string& input)
{
    IO::Dim3::read<TBASEMESH>(this, input);
}

T_TYPENAME
void TBASEMESH::write(const std::string& output)
{
    IO::Dim3::write<TBASEMESH>(this, output);
}

T_TYPENAME
void TBASEMESH::load(const std::vector<CPoint>& vertices, 
                     const std::vector<std::vector<int>>& volumes)
{
    std::map<int, CPoint> vert_id_point;
    std::map<int, std::vector<int>> volume_id_vids;

    for (int i = 0; i < vertices.size(); ++i)
        vert_id_point.insert(std::make_pair(i+1, vertices[i]));
    
    for (int i = 0; i < volumes.size(); ++i)
    {
        auto vids = volumes[i];
        for (int j = 0; j < vids.size(); ++j)
            vids[j] += 1;
        volume_id_vids.insert(std::make_pair(i + 1, vids));
    }

    load(vert_id_point, volume_id_vids);
}

T_TYPENAME
void TBASEMESH::load(const std::map<int, CPoint>& vert_id_point, 
                     const std::map<int, std::vector<int>>& volume_id_vids)
{
    // clear mesh
    this->unload();

    // add vertices
    for (auto& kv : vert_id_point)
    {
        int vid = kv.first;
        CVertex* pV = create_vertex(vid);
        pV->point() = kv.second;
    }

    // add volumes
    for (auto& kv : volume_id_vids)
    {
        int volid = kv.first;
        std::vector<int> vids = kv.second;
        create_volume(vids, volid);
    }

    // post processing
    _post_processing();
}

T_TYPENAME
void TBASEMESH::load_attributes(
    const std::map<int, std::string>& vert_id_str,
    const std::map<int, std::string>& volume_id_str,
    const std::vector<std::tuple<int, int, std::string>>& edge_attrs)
{
    for (const auto& id_str : vert_id_str)
    {
        CVertex* pV = id_vertex(id_str.first);
        pV->string() = id_str.second;

        pV->from_string();
    }

    for (const auto& id_str : volume_id_str)
    {
        CVolume* pVol = id_volume(id_str.first);
        pVol->string() = id_str.second;

        pVol->from_string();
    }

    for (const auto& edge : edge_attrs)
    {
        int vid0 = std::get<0>(edge);
        int vid1 = std::get<1>(edge);
        CVertex* pV0 = id_vertex(vid0);
        CVertex* pV1 = id_vertex(vid1);

        CEdge* pE = vertex_edge(pV0, pV1);
        pE->string() = std::get<2>(edge);

        pE->from_string();
    }
}

T_TYPENAME
tVertex* TBASEMESH::create_vertex(int vid) 
{
#ifdef USING_MEMORY_POOL
    CVertex* pV = m_vertex_pool.newElement();
#else
    CVertex* pV = new CVertex;
#endif
    pV->id() = vid;
    m_vertices.push_back(pV);
    m_map_vertex.insert(std::make_pair(vid, pV));

    return pV;
}

T_TYPENAME
tDart* TBASEMESH::create_volume(const std::vector<int>& vert_ids, int volid)
{
    // add volume
#ifdef USING_MEMORY_POOL
    CVolume* pVol = m_volume_pool.newElement();
#else
    CVolume* pVol = new CVolume;
#endif
    pVol->id() = volid;
    m_volumes.push_back(pVol);
    m_map_volume.insert(std::make_pair(volid, pVol));

    // add faces
    static int tbl_tet_face[4][3] = {{0, 1, 2}, {0, 2, 3}, {0, 3, 1}, {1, 3, 2}};
    CDart* darts[4]; // the darts attached to the faces
    for (int i = 0; i < 4; ++i)
    {
        std::vector<int> face_verts(3);
        for (int j = 0; j < 3; ++j)
        {
            face_verts[j] = vert_ids[tbl_tet_face[i][j]];
        }
        darts[i] = create_face(pVol, face_verts);
    }

    // set dart for new volume
    pVol->dart() = darts[0];

    // link faces
    for (int i = 0; i < 4; ++i)
    {
        for (int j = i + 1; j < 4; ++j)
            link_faces(darts[i], darts[j]);
    }

    return D(pVol);
}

T_TYPENAME
tDart* TBASEMESH::create_face(CVolume* pVol, const std::vector<int>& indices)
{
    // add face
    auto result = m_map_face_keys.find(&indices[0]);
    bool found = result != m_map_face_keys.end();

    CFace* pF = NULL;
    if (!found) // not found
    {
#ifdef USING_MEMORY_POOL
        pF = m_face_pool.newElement();
#else
        pF = new CFace;
#endif
        m_faces.push_back(pF);

        m_map_face_keys.insert(std::make_pair(&indices[0], pF));
    }
    else
        pF = result->second;

    // add edges attached on the face ccwly
    static int tbl_face_edge[3][2] = {{0, 1}, {1, 2}, {2, 0}};
    std::vector<int> edge_verts(2);
    std::vector<CDart*> darts; // darts on the edges
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 2; ++j)
            edge_verts[j] = indices[tbl_face_edge[i][j]];
        CDart* e = create_edge(pVol, pF, edge_verts);
        darts.push_back(e);
    }

    // set dart for new face
    if (!found)
        pF->dart() = darts[0];

    // link the edges
    for (int i = 0; i < darts.size(); ++i)
    {
        CDart* d1 = darts[i];
        CDart* d2 = darts[(i + 1) % darts.size()];
        link_edges(d1, d2);
    }

    // if the face is shared by two volumes, then link them.
    if (found)
    {
        link_volumes(D(pF), darts[0]);
    }

    // reutrn the new added dart even if there is no new added face
    return darts[0];
}

T_TYPENAME
tDart* TBASEMESH::create_edge(CVolume* pVol, CFace* pF, const std::vector<int>& indices)
{
    // add edge
    auto result = m_map_edge_keys.find(&indices[0]);
    bool found = result != m_map_edge_keys.end();
    
    CEdge* pE = NULL;
    if (!found) // not found
    {
#ifdef USING_MEMORY_POOL
        pE = m_edge_pool.newElement();
#else
        pE = new CEdge;
#endif
        m_edges.push_back(pE);

        m_map_edge_keys.insert(std::make_pair(&indices[0], pE));
    }
    else
        pE = result->second;

    // add new dart
    CVertex* pEnd = id_vertex(indices[1]);
#ifdef USING_MEMORY_POOL
    CDart* pD = m_dart_pool.newElement();
#else
    CDart* pD = new CDart;
#endif
    pD->cell(0) = pEnd;
    pD->cell(1) = pE;
    pD->cell(2) = pF;
    pD->cell(3) = pVol;
    m_darts.push_back(pD);

    // assign a dart to the target vertex in the situation of ccwly
    if (D(pEnd) == NULL) // not assigned yet
        pEnd->dart() = pD;

    // set dart for the new edge
    if (!found)
        pE->dart() = pD;

    // in the situation of combinatorial map
    // we do not need to link_vertices (set beta0)

    return pD;
}

T_TYPENAME
void TBASEMESH::link_faces(CDart* d1, CDart* d2)
{
    for (int i = 0; i < 3; ++i) // #edges of f1 = 3
    {
        for (int j = 0; j < 3; ++j) // #edges of f2 = 3
        {
            if (d1->cell(1) == d2->cell(1))
            {
                d1->beta(2) = d2;
                d2->beta(2) = d1;

                // We assume all faces are convex, so only one edge is shared
                // by f1 and f2. But it is not ture if there exists concave faces.
                return;
            }
            d2 = beta(1, d2); // beta1 has been set in link_edges
        }
        d1 = beta(1, d1);
    }
    printf("[ERROR] Should found!\n");
    exit(EXIT_FAILURE);
}

T_TYPENAME
void TBASEMESH::link_edges(CDart* d1, CDart* d2) { d1->beta(1) = d2; }

T_TYPENAME
void TBASEMESH::link_volumes(CDart* d1, CDart* d2)
{
    int found = 0;
    for (int i = 0; i < 3; ++i) // #edges of the shared face = 3
    {
        for (int j = 0; j < 3; ++j) // #edges of the shared face = 3
        {
            if (d1->cell(1) == d2->cell(1))
            {
                d1->beta(3) = d2;
                d2->beta(3) = d1;
                ++found;
            }
            d2 = beta(1, d2);
        }
        d1 = beta(1, d1);
    }

    if (found != 3) // #edges of the shared face = 3
    {
        printf("[ERROR] When we link volumes, there should be three pairs of darts! (%d)\n", found);
        exit(EXIT_FAILURE);
    }
}

T_TYPENAME
tVertex* TBASEMESH::dart_source(CDart* dart) 
{
    CDart* d = dart;
    while (true)
    {
        if (beta(1, d) != dart)
            d = beta(1, d);
        else
            return C0(d);
    }
}

T_TYPENAME
tDart* TBASEMESH::dart_prev(CDart* dart)
{
    CDart* d = dart;
    while (true)
    {
        if (beta(1, d) != dart)
            d = beta(1, d);
        else
            return d;
    }
}

T_TYPENAME
tEdge* TBASEMESH::vertex_edge(CVertex* v0, CVertex* v1) 
{
    std::vector<CDart*> darts = vertex_incident_darts(v1);
    for (auto pD : darts)
    {
        if (dart_source(pD) == v0)
            return C1(pD);
    }
    return NULL;
}

T_TYPENAME
tVertex* TBASEMESH::edge_vertex(CEdge* edge, int index)
{
    assert(index == 0 || index == 1);

    if (index == 0)
        return C0(D(edge));
    else if (index == 1)
        return C0(beta(2, D(edge)));

    return NULL;
}

T_TYPENAME
std::vector<tDart*> TBASEMESH::vertex_incident_darts(CVertex* vertex)
{
    std::vector<CDart*> darts;

    std::queue<CVolume*> Q;
    std::vector<CVolume*> volumes;

    CVolume* pVol = C3(D(vertex)); // vertex->dart()->volume();
    Q.push(pVol);
    volumes.push_back(pVol);

    while (!Q.empty())
    {
        pVol = Q.front();
        Q.pop();

        std::vector<CDart*> _darts = vertex_incident_darts(vertex, pVol);
        for (auto d : _darts)
        {
            darts.push_back(d);

            CDart* pD = beta(3, d);
            pVol = pD != NULL ? C3(pD) : NULL;

            if (pVol != NULL && std::find(volumes.begin(), volumes.end(), pVol) == volumes.end())
            {
                Q.push(pVol);
                volumes.push_back(pVol);
            }
        }
    }

    return darts;
}

T_TYPENAME
std::vector<tDart*> TBASEMESH::vertex_incident_darts(CVertex* vertex, CVolume* volume)
{
    // CAUTION:
    //   Only be valid on tetrahedron
    std::vector<CDart*> result;

    CDart* dart_start = NULL;
    if (C3(D(vertex)) == volume) // vertex->dart()->volume() 
        dart_start = D(vertex);
    else
    {
        auto darts = C32(volume);
        bool found = false;
        for (auto& d : darts)
        {
            for (int i = 0; i < 3; ++i)
            {
                if ((CVertex*)d->cell(0) == vertex)
                {
                    dart_start = d;
                    found = true;
                    break;
                }
                d = beta(1, d);
            }
            if (found)
                break;
        }
    }

    for (int i = 0; i < 3; ++i)
    {
        result.push_back(dart_start);
        dart_start = beta(2, beta(1, dart_start));
    }

    return result;
}

T_TYPENAME
std::vector<tDart*> TBASEMESH::C32(CVolume* volume)
{
    // CAUTION: only work on tet
    std::vector<CDart*> result;
    CDart* pD = D(volume);
    for (int i = 0; i < 3; ++i)
    {
        result.push_back(pD);
        pD = beta(2, beta(1, pD));
    }
    pD = beta(2, beta(1, beta(1, pD)));
    result.push_back(pD);

    return result;
}

T_TYPENAME
bool TBASEMESH::boundary(CVertex* v) 
{
    //TODO
    return false;
}

T_TYPENAME
void TBASEMESH::_post_processing()
{
    // 1. set boundary darts on boundary edges
    CEdge* pE;
    for (CDart* pD : m_darts)
    {
        if (pD->boundary())
        {
            pE = C1(pD);
            pE->dart() = pD;
        }
    }
}

} // namespace DartLib

#undef T_TYPENAME
#undef TBASEMESH
#endif // !_DARTLIB_MESH_3D_H_
