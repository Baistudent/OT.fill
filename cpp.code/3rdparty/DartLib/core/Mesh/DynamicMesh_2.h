#ifndef _DARTLIB_DYNAMIC_MESH_2D_H_
#define _DARTLIB_DYNAMIC_MESH_2D_H_

#include <queue>
#include <vector>

#include "BaseMesh_2.h"

#define T_TYPENAME template <typename tVertex, typename tEdge, typename tFace, typename tDart>
#define T_DYNAMIC_MESH TDynamicMesh_2<tVertex, tEdge, tFace, tDart>
#define T_BASE typename TBaseMesh_2<tVertex, tEdge, tFace, tDart>

namespace DartLib
{

T_TYPENAME
class TDynamicMesh_2 : public TBaseMesh_2<tVertex, tEdge, tFace, tDart>
{
  public:
    using T_BASE::CDart;
    using T_BASE::CEdge;
    using T_BASE::CFace;
    using T_BASE::CVertex;

    using T_BASE::CBoundary;
    using T_BASE::CLoop;

    using T_BASE::DartIterator;
    using T_BASE::EdgeIterator;
    using T_BASE::FaceIterator;
    using T_BASE::VertexIterator;

    using T_BASE::VertexEdgeIterator;
    using T_BASE::VertexFaceIterator;
    using T_BASE::VertexInDartIterator;
    using T_BASE::VertexVertexIterator;

    using T_BASE::FaceVertexIterator;
    using T_BASE::FaceEdgeIterator;
    using T_BASE::FaceDartIterator;
    

    /*===============================================================
                          Insert/Delete operation
    ===============================================================*/
 
  public:
    /*!
     *  Insert a vetex, it is the same with create_vertex.
     *  \param [in] vid: the vertex id will be assigned to the new vertex
     *  \return the pointer of the created vertex
     */
    tVertex* insert_vertex(int vid);
    
    /*!
     *  Insert a new face, the difference between create_face and 
     *    insert_face is that the latter will modify the dart linking
     *    to the boundary vertex to a boundary dart.
     *  \return the pointer of the created face
     */
    tFace* insert_face(const std::vector<int>& vert_ids, int fid);

    /*!
     *  Remove a vertex.
     *    It will remove all surrounding faces and then release itself.
     *  \param [in] pV: the input vertex
     */
    void remove_vertex(CVertex* pV);

    /*!
     *  Remove an edge.
     *    It will remove all surrounding faces and then release itself.
     *  \param [in] pE: the input edge
     */
    void remove_edge(CEdge* pE);

    /*!
     *  Remove a face, this is a key operation.
     *    It will remove a face and maybe release some vertices and edges
     *    if necessary, then release itself.
     *  \param [in]                    pF: the to-be-removed face.
     *  \param [in] remove_isolated_verts: whether to remove the isolated vertices.
     *  \return false if the mesh becomes a non-manifold after removing the face,
     *          or true if not.
     */
    bool remove_face(CFace* pF, bool remove_isolated_verts = true);

    /*!
     *  Remove a list of faces.
     *    Caution: we assume that the list of faces is a topological disk.
     *  \param [in] volumes: the input volumes
     */
    void remove_faces(std::vector<CFace*>& faces, bool remove_isolated_verts = true);

  protected:
    /*!
     *  Release a vertex.
     *    It only releases its memory space.
     *  \param [in] pV: the input vertex
     */
    void release_vertex(CVertex* pV);

    /*!
     *  Release an edge.
     *    It only releases its memory space.
     *  \param [in] pE: the input edge
     */
    void release_edge(CEdge* pE);

    /*!
     *  Release a face.
     *    It only releases its memory space.
     *  \param [in] pF: the input face
     */
    void release_face(CFace* pF);

    /*!
     *  Release a dart.
     *    It modifies the beta_2 function which the dart affects it,
     *    then releases its memory space.
     * 
     *  NOTE: We do not release a dart solely, often we remove a face
     *    and need to release all dart on the face.
     * 
     *  \param [in] pD: the input dart
     */
    void release_dart(CDart* pD);

    /*===============================================================
                          Dynamic operation
    ===============================================================*/
  public:
    /*!
     *  Edge swap operation for triangle mesh.
     *  \param [in] pE: the input edge which will be flipped.
     *  \return the edge pointer of the swapped edge.
     */
    tEdge* edge_swap(CEdge* pE);
    
    //CVertex edge_collapse(CEdge* pE);
    //CEdge* vertex_split(CVertex* pV);
    
    /*!
     *  Edge split operation for triangle mesh
     *  \param [in] pE: the input edge which will be splited.
     *  \return the vertex pointer of the added vertex which lies on the edge.
     */
    tVertex* edge_split(CEdge* pE);

    /*!
     *  Face split operation for polygon mesh.
     *  \param [in] pF: the input face which will be splited.
     *  \return the vertex pointer of the added vertex which lies inside the face.
     */
    tVertex* face_split(CFace* pF);
};

T_TYPENAME
tVertex* T_DYNAMIC_MESH::insert_vertex(int vid)
{
    return this->create_vertex(vid);
}

T_TYPENAME
tFace* T_DYNAMIC_MESH::insert_face(const std::vector<int>& indices, int fid)
{
    CDart* pD = this->create_face(indices, fid);

    // TODO, modify the dart linking to the boundary vertex to a boundary dart
    // 

    return this->dart_face(pD);
}

T_TYPENAME
void T_DYNAMIC_MESH::remove_vertex(CVertex* pV)
{
    for (VertexFaceIterator vfiter(pV); !vfiter.end(); ++vfiter)
    {
        CFace* pF = *vfiter;
        remove_face(pF);
    }

    release_vertex(pV); // not necessary
}

T_TYPENAME
void T_DYNAMIC_MESH::remove_edge(CEdge* pE)
{
    CFace* faces[2] = {NULL, NULL};
    for (int i = 0; i < 2; ++i)
        faces[i] = this->edge_face(pE, i);

    for (int i = 0; i < 2; ++i)
        remove_face(faces[i]);

    release_edge(pE); // not necessary
}

T_TYPENAME
bool T_DYNAMIC_MESH::remove_face(CFace* pF, bool remove_isolated_verts)
{
    if (pF == NULL)
        return true;

    // check non-manifold, if we remove face ABC, then there will 
    //   form non-manifold.
    /* 
            A------C
            / \  / \
           /   \/   \
           ----B-----       
    */
    std::vector<CVertex*> vs;
    for (FaceVertexIterator fviter(pF); !fviter.end(); ++fviter)
        vs.push_back(*fviter);

    for (int i = 0; i < vs.size(); ++i)
    {
        CVertex* b = vs[i];
        if (this->boundary(b))
        {
            CVertex* a = vs[(i - 1 + vs.size()) % vs.size()];
            CVertex* c = vs[(i + 1) % vs.size()];
            if (!this->boundary(this->vertex_dart(a, b)) &&
                !this->boundary(this->vertex_dart(b, c)))
                return false;
        }
    }

    CDart* pD0 = this->face_dart(pF);
    CDart* pD = pD0;
    std::vector<CDart*> darts;
    std::vector<CEdge*> edges;
    std::vector<CVertex*> verts;
    do
    {
        darts.push_back(pD);
        edges.push_back(this->dart_edge(pD));
        verts.push_back(this->dart_target(pD));
        pD = this->dart_next(pD);
    } while (pD != pD0);

    for (int i = 0; i < edges.size(); ++i)
    {
        CEdge* pE = edges[i];
        if (!this->boundary(darts[i]) && pE->dart() == darts[i])
            pE->dart() = this->dart_sym(darts[i]); // edge_dart(pE, 1)

        if (this->boundary(darts[i]))
            release_edge(pE);
    }

    for (auto v : verts)
    {
        bool b = this->boundary(v);
        int valance = 0;
        for (VertexFaceIterator vfiter(v); !vfiter.end(); ++vfiter)
            ++valance;

        if (std::find(darts.begin(), darts.end(), v->dart()) != darts.end())
        {
            v->dart() = v->dart()->beta(1)->beta(2);
        }
        else
        {
            if (!b)
            {
                for (auto d : darts)
                {
                    if (this->dart_target(d) == v)
                    {
                        v->dart() = this->dart_sym(this->dart_next(d));
                        break;
                    }
                }
            }
        }

        // if (v->dart() == NULL)
        if (valance == 1 && remove_isolated_verts)
            release_vertex(v);
    }

    release_face(pF);

    for (auto d : darts)
    {
        release_dart(d);
    }

    return true;
}

T_TYPENAME
void T_DYNAMIC_MESH::remove_faces(std::vector<CFace*>& removed_faces, bool remove_isolated_verts)
{
    std::vector<CEdge*>   boundary_edges; // boundary of removed faces
    std::map<int, CFace*> map_id_face;

    std::set<CVertex*>  removed_vertices;
    std::vector<CEdge*> removed_edges;
    std::vector<CDart*> removed_darts;

    // collect removed edges and boundary edges
    for (auto pF : removed_faces)
    {
        map_id_face.insert(std::make_pair(pF->id(), pF));

        for (FaceEdgeIterator feit(pF); !feit.end(); ++feit)
        {
            CEdge* pE = *feit;
            auto eit = std::find(boundary_edges.begin(), boundary_edges.end(), pE);
            if (eit == boundary_edges.end())
            {
                boundary_edges.push_back(pE);
                if (this->boundary(pE))
                    removed_edges.push_back(pE);
            }
            else
            {
                boundary_edges.erase(eit);
                removed_edges.push_back(pE);
            }
        }
    }

    // collect removed darts, vertices
    for (auto pF : removed_faces)
    {
        for (FaceDartIterator fdit(pF); !fdit.end(); ++fdit)
        {
            auto pD = *fdit;
            removed_darts.push_back(pD);
        }

        for (FaceVertexIterator fvit(pF); !fvit.end(); ++fvit)
        {
            auto pV = *fvit;
            bool removed = true;

            for (VertexFaceIterator vfit(pV); !vfit.end(); ++vfit)
            {
                int fid = (*vfit)->id();
                if (map_id_face.find(fid) == map_id_face.end())
                {
                    removed = false;
                    break;
                }
            }

            if (removed) removed_vertices.insert(pV);
        }
    }

    // modify
    for (auto pE : boundary_edges)
    {
        CDart* pD = this->edge_dart(pE, 0);
        
        int fid = this->dart_face(pD)->id();
        if (map_id_face.find(fid) != map_id_face.end()) // the face in removed_faces
        {
            pD = this->beta(2, pD);
        }
        
        if (pD != NULL) // TODO: pD != NULL, why?
        {
            CVertex* pV = this->dart_target(pD);
            pV->dart() = pD;
            pE->dart() = pD;
            pD->beta(2) = NULL;
        }
    }

    // delete
    if (remove_isolated_verts)
    {
        for (auto pV : removed_vertices)
            release_vertex(pV);
    }
    for (auto pE : removed_edges)    release_edge(pE);
    for (auto pF : removed_faces)    release_face(pF);
    for (auto pD : removed_darts)    release_dart(pD);
}

T_TYPENAME
void T_DYNAMIC_MESH::release_vertex(CVertex* pV)
{
    this->m_vertices.remove(pV);
    this->m_map_vertex.erase(pV->id());

    delete pV;
    pV = NULL;
}

T_TYPENAME
void T_DYNAMIC_MESH::release_edge(CEdge* pE)
{
    if (pE == NULL)
        return;

    int vid0 = this->edge_vertex(pE, 0)->id();
    int vid1 = this->edge_vertex(pE, 1)->id();
    std::vector<int> indices = {vid0, vid1};
    this->m_map_edge_keys.erase(&indices[0]);

    this->m_edges.remove(pE);

    delete pE;
    pE = NULL;
}

T_TYPENAME
void T_DYNAMIC_MESH::release_face(CFace* pF)
{
    this->m_faces.remove(pF);
    this->m_map_face.erase(pF->id());

    delete pF;
    pF = NULL;
}

T_TYPENAME
void T_DYNAMIC_MESH::release_dart(CDart* pD)
{
    if (pD->beta(2) != NULL) // not on boundary
    {
        CDart* pSymDart = this->dart_sym(pD);
        pSymDart->beta(2) = NULL;
    }

    // necessary, TODO strange!
    for (int i = 0; i <= 2; ++i)
        pD->cell(i) = NULL;
    for (int i = 1; i <= 2; ++i)
        pD->beta(i) = NULL;

    this->m_darts.remove(pD);
    delete pD;
    pD = NULL;
}

T_TYPENAME
tEdge* T_DYNAMIC_MESH::edge_swap(CEdge* pE)
{
    assert(!this->boundary(pE));

    CDart* pD[6];
    for (int i = 0; i < 2; ++i)
    {
        pD[i * 3] = this->edge_dart(pE, i);
        pD[i * 3 + 1] = this->dart_next(pD[i * 3]);
        pD[i * 3 + 2] = this->dart_next(pD[i * 3 + 1]);
    }

    CVertex* pV[2];
    pV[0] = this->dart_target(pD[0]);
    pV[1] = this->dart_target(pD[3]);

    CFace* pF[2];
    pF[0] = this->dart_face(pD[0]);
    pF[1] = this->dart_face(pD[3]);

    // modify beta function on dart
    pD[0]->beta(1) = pD[2];
    pD[2]->beta(1) = pD[4];
    pD[4]->beta(1) = pD[0];
    
    pD[3]->beta(1) = pD[5];
    pD[5]->beta(1) = pD[1];
    pD[1]->beta(1) = pD[3];

    // modify 0-cell on dart
    pD[0]->cell(0) = pD[1]->cell(0);
    pD[3]->cell(0) = pD[4]->cell(0);

    // modify 2-cell on dart
    pD[1]->cell(2) = pF[1];
    pD[4]->cell(2) = pF[0];

    // modify dart on 0-cell
    if (pV[0]->dart() == pD[0])
        pV[0]->dart() = pD[5];
    if (pV[1]->dart() == pD[3])
        pV[1]->dart() = pD[2];

    // modify dart on 2-cell
    if (pF[0]->dart() == pD[1])
        pF[0]->dart() = pD[0];
    if (pF[1]->dart() == pD[4])
        pF[1]->dart() = pD[3];

    return pE;
}

T_TYPENAME
tVertex* T_DYNAMIC_MESH::edge_split(CEdge* pE) 
{
    bool isBoundary = this->boundary(pE);

    int vid = 0, fid = 0;
    for (VertexIterator vit(this); !vit.end(); ++vit)
    {
        CVertex* pV = *vit;
        vid = pV->id() > vid ? pV->id() : vid;
    }
    CVertex* pV = this->create_vertex(++vid);

    for (FaceIterator fit(this); !fit.end(); ++fit)
    {
        CFace* f = *fit;
        fid = f->id() > fid ? f->id() : fid;
    }

    std::vector<CFace*> attached_faces(2);
    attached_faces[0] = this->edge_face(pE, 0);
    attached_faces[1] = this->edge_face(pE, 1);

    int vids[5];
    vids[0] = vid;

    CDart* pD = this->face_dart(attached_faces[0]);
    vids[1] = this->dart_target(pD)->id();
    pD = this->dart_next(pD);
    vids[2] = this->dart_target(pD)->id();
    pD = this->dart_next(pD);
    vids[3] = this->dart_target(pD)->id();
    if (!isBoundary)
    {
        pD = this->face_dart(attached_faces[1]);
        pD = this->dart_next(pD);
        vids[4] = this->dart_target(pD)->id();
    }

    bool success = this->remove_face(attached_faces[0], false);
    if (success && !isBoundary)
    {
        success = this->remove_face(attached_faces[1], false);
    }

    if (success)
    {
        std::vector<int> face_vids(3);
        face_vids[0] = vids[0], face_vids[1] = vids[1], face_vids[2] = vids[2];
        this->create_face(face_vids, ++fid);
        face_vids[0] = vids[0], face_vids[1] = vids[2], face_vids[2] = vids[3];
        this->create_face(face_vids, ++fid);
        if (!isBoundary)
        {
            face_vids[0] = vids[0], face_vids[1] = vids[3], face_vids[2] = vids[4];
            this->create_face(face_vids, ++fid);
            face_vids[0] = vids[0], face_vids[1] = vids[4], face_vids[2] = vids[1];
            this->create_face(face_vids, ++fid);
        }

        return pV;
    }

    return NULL;
}

T_TYPENAME
tVertex* T_DYNAMIC_MESH::face_split(CFace* pF) 
{
    int vid = 0, fid = 0;
    for (VertexIterator vit(this); !vit.end(); ++vit)
    {
        CVertex* pV = *vit;
        vid = pV->id() > vid ? pV->id() : vid;
    }
    CVertex* pV = this->create_vertex(++vid);

    for (FaceIterator fit(this); !fit.end(); ++fit)
    {
        CFace* f = *fit;
        fid = f->id() > fid ? f->id() : fid;
    }

    int sid, tid;
    std::vector<std::vector<int>> faces;
    std::vector<int> face_verts(3);
    for (FaceDartIterator fdit(pF); !fdit.end(); ++fdit)
    {
        CDart* pD = *fdit;
        sid = this->dart_source(pD)->id();
        tid = this->dart_target(pD)->id();
        face_verts[0] = sid, face_verts[1] = tid, face_verts[2] = vid;
        faces.push_back(face_verts);
    }

    //std::vector<CFace*> flist;
    //flist.push_back(pF);
    //printf("split face id: %d\n", pF->id());
    bool success = this->remove_face(pF, false);
    
    if (success)
    {
        for (auto& face : faces)
        {
            this->create_face(face, ++fid);
        }
        return pV;
    }

    return NULL;
}

} // namespace DartLib
#undef T_TYPENAME
#undef T_DYNAMIC_MESH
#undef T_BASE
#endif //! _DARTLIB_DYNAMIC_MESH_2D_H_
