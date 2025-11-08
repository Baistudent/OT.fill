#ifndef _DARTLIB_DYNAMIC_MESH_3D_H_
#define _DARTLIB_DYNAMIC_MESH_3D_H_

#include <queue>
#include <vector>
#include <list>

#include "BaseMesh_3.h"

#define T_TYPENAME template <typename tVertex, typename tEdge, typename tFace, typename tVolume, typename tDart>
#define T_DYNAMIC_MESH TDynamicMesh_3<tVertex, tEdge, tFace, tVolume, tDart>
#define T_BASE typename TBaseMesh_3<tVertex, tEdge, tFace, tVolume, tDart>

namespace DartLib
{

T_TYPENAME
class TDynamicMesh_3 : public TBaseMesh_3<tVertex, tEdge, tFace, tVolume, tDart>
{
    /*===============================================================
                          Insert/Delete operation
    ===============================================================*/

  public:
    using T_BASE::CDart;
    using T_BASE::CVertex;
    using T_BASE::CEdge;
    using T_BASE::CFace;
    using T_BASE::CVolume;

    using T_BASE::CBoundary;
    //using T_BASE::CLoop;

    using T_BASE::DartIterator;
    using T_BASE::VertexIterator;
    using T_BASE::EdgeIterator;
    using T_BASE::FaceIterator;
    using T_BASE::VolumeIterator;

    using T_BASE::VertexVertexIterator;
    using T_BASE::VertexEdgeIterator;
    using T_BASE::VertexFaceIterator;
    using T_BASE::VertexVolumeIterator;

    using T_BASE::EdgeVolumeIterator;
    using T_BASE::EdgeFaceIterator;
    
    using T_BASE::FaceDartIterator;
    using T_BASE::FaceVertexIterator;
    using T_BASE::FaceEdgeIterator;

    using T_BASE::VolumeVertexIterator;
    using T_BASE::VolumeEdgeIterator;
    using T_BASE::VolumeFaceIterator;

    /*!
     *  Insert a vetex, it is the same with create_vertex.
     *  \param [in] vid: the vertex id will be assigned to the new vertex
     *  \return the pointer of the created vertex
     */
    tVertex* insert_vertex(int vid);

    /*!
     *  Remove a vertex.
     *    It will remove all surrounding faces and then release itself.
     *  \param [in] pV: the input vertex
     */
    void remove_vertex(CVertex* pV);

    /*£¡
     *  Remove a volume.
     *    Caution: you need to ensure that there is no non-manifold after removing.
     *  \param [in] pVol: the input volume
     */
    void remove_volume(CVolume* pVol);

    /*! 
     *  Remove a list of volumes.
     *    Caution: we assume that the list of volumes is a topological sphere.
     *  \param [in] volumes: the input volumes
     */
    void remove_volumes(std::vector<CVolume*> & volumes);
  protected:

    /*!
     *  Release a vertex.
     *    It only releases its memory space.
     *  \param [in] pV: the input vertex
     */
    void _release_vertex(CVertex* pV);

    void _release_edge(CEdge* pE);

    void _release_face(CFace* pF);

    void _release_volume(CVolume* pVol);

    void _release_dart(CDart* pD);

    /*===============================================================
                          Dynamic operation
    ===============================================================*/
  public:
    

};

T_TYPENAME
tVertex* T_DYNAMIC_MESH::insert_vertex(int vid) { return this->create_vertex(vid); }

T_TYPENAME
void T_DYNAMIC_MESH::remove_vertex(CVertex* pV)
{
    for (T_BASE::VertexFaceIterator vfiter(pV); !vfiter.end(); ++vfiter)
    {
        CFace* pF = *vfiter;
        remove_face(pF);
    }

    release_vertex(pV); // not necessary
}
T_TYPENAME
void T_DYNAMIC_MESH::remove_volume(CVolume* pVol) 
{
    if (pVol == NULL) return;
    
    // check non-manifold


    // collect the information on the volume
    #define _NORMAL_ '0'
    #define _DELETE_ '1'
    std::vector<CDart*>        darts;
    std::map<CVertex*,char> vert_tag;
    std::map<CEdge*,  char> edge_tag;
    std::map<CFace*,  char> face_tag;
    for (VolumeFaceIterator vfit(pVol); !vfit.end(); ++vfit)
    {
        CFace* pF = *vfit;
        face_tag.insert(std::make_pair(pF, _NORMAL_));
        for (FaceDartIterator fdit(pF); !fdit.end(); ++fdit)
        {
            CDart* pD = *fdit;
            darts.push_back(pD);
            edge_tag.insert(std::make_pair(this->dart_edge(pD), _NORMAL_));
        }
    }

    for (VolumeVertexIterator vvit(pVol); !vvit.end(); ++vvit)
    {
        CVertex* pV = *vvit;
        vert_tag.insert(std::make_pair(pV, _NORMAL_));
    }

    // find all elements which will be deleted
    for (auto& ft : face_tag)
    {
        CFace* pF = ft.first;
        if (this->boundary(pF))
            ft.second = _DELETE_;
    }

    for (auto& et : edge_tag)
    {
        CEdge* pE = et.first;
        int nVol = 0;
        for (EdgeVolumeIterator evit(pE); !evit.end(); ++evit)
            ++nVol;
        if (nVol == 1)
            et.second = _DELETE_;
    }

    for (auto& vt : vert_tag)
    {
        CVertex* pV = vt.first;
        int nVol = 0;
        for (VertexVolumeIterator vvit(this, pV); !vvit.end(); ++vvit)
            ++nVol;
        if (nVol == 1)
            vt.second = _DELETE_;
    }

    // modify the beta_3 funcion
    for (auto pD : darts)
    {
        pD = this->beta(3, pD);
        if (pD != NULL)
        {
            CVertex* pV = this->dart_target(pD);
            CEdge*   pE = this->dart_edge(pD);
            CFace*   pF = this->dart_face(pD);

            pV->dart() = pD;
            pE->dart() = pD;
            pF->dart() = pD;
            pD->beta(3) = NULL;
        }
    }

    // release the elements, be careful of the order.
    for (auto et : edge_tag)
    {
        if (et.second == _DELETE_)
            _release_edge(et.first);
    }

    for (auto ft : face_tag)
    {
        if (ft.second == _DELETE_)
            _release_face(ft.first);
    }

    for (auto vt : vert_tag)
    {
        if (vt.second == _DELETE_)
            _release_vertex(vt.first);
    }

    _release_volume(pVol);
    
    for (auto pD : darts)
        _release_dart(pD);

    #undef _NORMAL_
    #undef _DELETE_
}

T_TYPENAME
void T_DYNAMIC_MESH::remove_volumes(std::vector<CVolume*>& volumes)
{
    std::vector<CFace*>     boundary_faces; // boundary of removed volumes
    std::map<int, CVolume*> map_id_volume;
    
    std::set<CVertex *>     removed_vertices;
    std::set<CEdge   *>     removed_edges;
    std::vector<CFace*>     removed_faces;
    std::vector<CDart*>     removed_darts;

    // collect removed faces and boundary faces
    for (auto vol : volumes)
    {
        map_id_volume.insert(std::make_pair(vol->id(), vol));

        for (VolumeFaceIterator vfit(vol); !vfit.end(); ++vfit)
        {
            CFace* pF = *vfit;
            auto fit = std::find(boundary_faces.begin(), boundary_faces.end(), pF);
            if (fit == boundary_faces.end())
            {
                boundary_faces.push_back(pF);
                if (this->boundary(pF))
                    removed_faces.push_back(pF);
            }
            else
            {
                boundary_faces.erase(fit);
                removed_faces.push_back(pF);
            }
        }
    }

    // collect removed darts, vertices, edges
    for (auto vol : volumes)
    {
        for (VolumeFaceIterator vfit(vol); !vfit.end(); ++vfit)
        {
            CFace* pF = *vfit;
            for (FaceDartIterator fdit(pF); !fdit.end(); ++fdit)
            {
                CDart* pD = *fdit;

                if (this->dart_volume(pD) != vol)
                    pD = this->beta(3, pD);

                removed_darts.push_back(pD);
            }

            for (FaceEdgeIterator feit(pF); !feit.end(); ++feit)
            {
                CEdge* pE = *feit;
                bool removed = true;

                //auto fit = std::find(boundary_faces.begin(), boundary_faces.end(), pF);
                //if (fit != boundary_faces.end())
                //{
                    for (EdgeVolumeIterator evit(pE); !evit.end(); ++evit)
                    {
                        int vol_id = (*evit)->id();
                        if (map_id_volume.find(vol_id) == map_id_volume.end())
                        {
                            removed = false;
                            break;
                        }
                    }
                //}

                if (removed) removed_edges.insert(pE);
            }

            for (FaceVertexIterator fvit(pF); !fvit.end(); ++fvit)
            {
                CVertex* pV = *fvit;
                bool removed = true;
                
                //auto fit = std::find(boundary_faces.begin(), boundary_faces.end(), pF);
                //if (fit != boundary_faces.end())
                //{
                    for (VertexVolumeIterator vvit(this, pV); !vvit.end(); ++vvit)
                    {
                        int vol_id = (*vvit)->id();
                        if (map_id_volume.find(vol_id) == map_id_volume.end())
                        {
                            removed = false;
                            break;
                        }
                    }
                //}

                if (removed) removed_vertices.insert(pV);
            }
        }
    }

    // modify
    for (auto pF : boundary_faces)
    {
        CDart* pD = NULL;
        for (FaceDartIterator fdit(pF); !fdit.end(); ++fdit)
        {
            pD = *fdit;
            int vol_id = this->dart_volume(pD)->id();
            if (map_id_volume.find(vol_id) != map_id_volume.end())// pD on removed volumes
            {
                pD = this->beta(3, pD);
            }

            CEdge*   pE = this->dart_edge(pD);
            CVertex* pV = this->dart_target(pD);
            pV->dart() = pD;
            pE->dart() = pD;
            pD->beta(3) = NULL;
        }
        pF->dart() = pD;
    }

    // delete
    assert(removed_darts.size() == 12 * volumes.size());

    for (auto pV : removed_vertices) _release_vertex(pV);
    for (auto pE : removed_edges)    _release_edge(pE);
    for (auto pF : removed_faces)    _release_face(pF);
    for (auto pVol : volumes)        _release_volume(pVol);
    for (auto pD : removed_darts)    _release_dart(pD);

    this->_post_processing();
}

T_TYPENAME
void T_DYNAMIC_MESH::_release_vertex(CVertex* pV)
{
    auto it = std::find(this->m_vertices.begin(), this->m_vertices.end(), pV);
    if (it != this->m_vertices.end())
        this->m_vertices.erase(it);
    this->m_map_vertex.erase(pV->id());

#ifdef USING_MEMORY_POOL
    this->m_vertex_pool.deleteElement(pV);
#else
    delete pV;
    pV = NULL;
#endif
}

T_TYPENAME
void T_DYNAMIC_MESH::_release_edge(CEdge* pE)
{
    if (pE == NULL)
        return;

    int vid0 = this->edge_vertex(pE, 0)->id();
    int vid1 = this->edge_vertex(pE, 1)->id();
    std::vector<int> indices = {vid0, vid1};
    this->m_map_edge_keys.erase(&indices[0]);

    auto it = std::find(this->m_edges.begin(), this->m_edges.end(), pE);
    if (it != this->m_edges.end())
        this->m_edges.erase(it);

#ifdef USING_MEMORY_POOL
    this->m_edge_pool.deleteElement(pE);
#else
    delete pE;
    pE = NULL;
#endif
}

T_TYPENAME
void T_DYNAMIC_MESH::_release_face(CFace* pF)
{
    std::vector<int> indices;
    for (FaceVertexIterator fvit(pF); !fvit.end(); ++fvit)
    {
        CVertex* pV = *fvit;
        indices.push_back(pV->id());
    }

    this->m_map_face_keys.erase(&indices[0]);
    
    auto it = std::find(this->m_faces.begin(), this->m_faces.end(), pF);
    if (it != this->m_faces.end())
        this->m_faces.erase(it);

#ifdef USING_MEMORY_POOL
    this->m_face_pool.deleteElement(pF);
#else
    delete pF;
    pF = NULL;
#endif
}


T_TYPENAME
void T_DYNAMIC_MESH::_release_volume(CVolume* pVol)
{
    auto it = std::find(this->m_volumes.begin(), this->m_volumes.end(), pVol);
    if (it != this->m_volumes.end())
        this->m_volumes.erase(it);
    this->m_map_volume.erase(pVol->id());

#ifdef USING_MEMORY_POOL
    this->m_volume_pool.deleteElement(pVol);
#else
    delete pVol;
    pVol = NULL;
#endif
}

T_TYPENAME
void T_DYNAMIC_MESH::_release_dart(CDart* pD)
{
    auto it = std::find(this->m_darts.begin(), this->m_darts.end(), pD);
    if (it != this->m_darts.end())
        this->m_darts.erase(it);

#ifdef USING_MEMORY_POOL
    this->m_dart_pool.deleteElement(pD);
#else
    delete pD;
    pD = NULL;
#endif
}

} // namespace DartLib
#undef T_TYPENAME
#undef T_DYNAMIC_MESH
#undef T_BASE
#endif //! _DARTLIB_DYNAMIC_MESH_3D_H_
