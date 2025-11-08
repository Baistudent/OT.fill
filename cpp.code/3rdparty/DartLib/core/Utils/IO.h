#ifndef _DARTLIB_IO_H_
#define _DARTLIB_IO_H_

#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <fstream>
#include <iostream>

#include "../Geometry/Point2.h"
#include "../Geometry/Point.h"
#include "../Parser/strutil.h"

#define MAX_LINE 1024

namespace DartLib
{
namespace IO
{

/*===================================================================
                    IO functions for 2-manifold
===================================================================*/

namespace Dim2
{
template <class M>
void read(M* pMesh, const std::string& input);

template <class M>
void read_m(M* pMesh, const std::string& input);

template <class M>
void read_obj(M* pMesh, const std::string& input);

template <class M>
void write(M* pMesh, const std::string& output);

template <class M>
void write_m(M* pMesh, const std::string& output);

template <class M>
void write_obj(M* pMesh, const std::string& output);


template <class M>
void read(M* pMesh, const std::string& input)
{
    if (strutil::endsWith(input, ".m"))
        read_m<M>(pMesh, input);
    else if (strutil::endsWith(input, ".obj"))
        read_obj<M>(pMesh, input);
    else
    {
        std::cerr << "Not support to read in " << input << "\n";
        return;
    }
}

template <class M>
void read_m(M* pMesh, const std::string& input)
{
    std::map<int, CPoint>    vert_id_point; // vid -> coordinate
    std::map<int, std::string> vert_id_str; // vid -> string

    std::map<int, std::vector<int>> face_id_vids; // fid -> vert_idx
    std::map<int, std::string>       face_id_str; // fid -> string

    std::vector<std::tuple<int, int, std::string>>   edge_attrs; //(vid1, vid2) -> string
    std::vector<std::tuple<int, int, std::string>> corner_attrs; //(vid,   fid) -> string

    std::fstream fs(input, std::fstream::in);
    if (fs.fail())
    {
        std::cerr << "Error in opening file " << input << "\n";
        return;
    }

    char buffer[MAX_LINE];
    int vid, fid;

    // 1. read data
    while (fs.getline(buffer, MAX_LINE))
    {
        std::string line(buffer);
        line = strutil::trim(line);

        strutil::Tokenizer stokenizer(line, " \r\n");

        stokenizer.nextToken();
        std::string token = stokenizer.getToken();

        if (token == "Vertex")
        {
            stokenizer.nextToken();
            token = stokenizer.getToken();
            vid = strutil::parseString<int>(token);

            CPoint p;
            for (int i = 0; i < 3; i++)
            {
                stokenizer.nextToken();
                token = stokenizer.getToken();
                p[i] = strutil::parseString<double>(token);
            }
            vert_id_point.insert(std::make_pair(vid, p));

            if (!stokenizer.nextToken("\t\r\n"))
                continue;
            token = stokenizer.getToken();

            int sp = (int) token.find("{");
            int ep = (int) token.find("}");

            if (sp >= 0 && ep >= 0)
            {
                std::string str = token.substr(sp + 1, ep - sp - 1);
                vert_id_str.insert(std::make_pair(vid, str));
            }
            continue;
        }

        if (token == "Face")
        {
            stokenizer.nextToken();
            token = stokenizer.getToken();
            fid = strutil::parseString<int>(token);

            std::vector<int> vert_ids;

            while (stokenizer.nextToken())
            {
                token = stokenizer.getToken();
                if (strutil::startsWith(token, "{"))
                    break;
                vid = strutil::parseString<int>(token);
                vert_ids.push_back(vid);
            }
            face_id_vids.insert(std::make_pair(fid, vert_ids));
            
            if (!stokenizer.nextToken("\t\r\n"))
                continue;
            token = stokenizer.getToken();

            token = line;
            int sp = (int) token.find("{");
            int ep = (int) token.find("}");

            if (sp >= 0 && ep >= 0)
            {
                std::string str = token.substr(sp + 1, ep - sp - 1);
                face_id_str.insert(std::make_pair(fid, str));
            }
            /*
            if( strutil::startsWith( token, "{" ) )
            {
                f->string() = strutil::trim( token, "{}" );
            }
            */
            continue;
        }

        // read in edge attributes
        if (token == "Edge")
        {
            stokenizer.nextToken();
            token = stokenizer.getToken();
            int id0 = strutil::parseString<int>(token);

            stokenizer.nextToken();
            token = stokenizer.getToken();
            int id1 = strutil::parseString<int>(token);

            if (!stokenizer.nextToken("\t\r\n"))
                continue;
            token = stokenizer.getToken();

            int sp = (int) token.find("{");
            int ep = (int) token.find("}");

            std::string str;
            if (sp >= 0 && ep >= 0)
            {
                str = token.substr(sp + 1, ep - sp - 1);
            }
            edge_attrs.push_back(std::make_tuple(id0, id1, str));

            continue;
        }

        // read in corner attributes
        if (token == "Corner")
        {
            stokenizer.nextToken();
            token = stokenizer.getToken();
            int vid = strutil::parseString<int>(token);

            stokenizer.nextToken();
            token = stokenizer.getToken();
            int fid = strutil::parseString<int>(token);

            if (!stokenizer.nextToken("\t\r\n"))
                continue;
            token = stokenizer.getToken();

            int sp = (int) token.find("{");
            int ep = (int) token.find("}");

            std::string str;
            if (sp >= 0 && ep >= 0)
            {
                str = token.substr(sp + 1, ep - sp - 1);
            }
            corner_attrs.push_back(std::make_tuple(vid, fid, str));

            continue;
        }
    }

    // 2. build mesh
    pMesh->load(vert_id_point, face_id_vids);

    // 3. read traits
    pMesh->load_attributes(vert_id_str, face_id_str, edge_attrs, corner_attrs);
}

template<class M>
void read_obj(M* pMesh, const std::string& input)
{
    typedef struct
    {
        int vid;
        int tid;
        int nid;
    } svertex;

    typedef struct
    {
        std::vector<svertex> verts;
    }sface;

    std::fstream fs(input, std::fstream::in);
    if (fs.fail())
    {
        fprintf(stderr, "Error is opening file %s\n", input.c_str());
        return;
    }

    char cmd[MAX_LINE];

    int  vid = 1;
    int  fid = 1;

    bool with_rgb = false;
    bool with_uv  = false;
    bool with_normal = false;

    std::map<int, CPoint>           vert_id_point;  // vid -> coordinate
    std::map<int, std::vector<int>> face_id_vids;   // fid -> vert_idx

    std::vector<CPoint>  rgbs;
    std::vector<CPoint2> uvs;
    std::vector<CPoint>  normals;
    std::vector<sface>   sfaces;

    // 1. read data
    while (fs.getline(cmd, MAX_LINE))
    {
        std::string line(cmd);
        line = strutil::trim(line);

        strutil::Tokenizer stokenizer(line, " \t\r\n");

        stokenizer.nextToken();
        std::string token = stokenizer.getToken();

        if (token == "v")
        {
            CPoint p;
            for (int i = 0; i < 3; i++)
            {
                stokenizer.nextToken();
                token = stokenizer.getToken();
                p[i] = strutil::parseString<double>(token);
            }

            vert_id_point.insert(std::make_pair(vid, p));

            if (stokenizer.nextToken())
            {
                with_rgb = true;
                CPoint rgb;
                for (int i = 0; i < 3; i++)
                {
                    token = stokenizer.getToken();
                    rgb[i] = strutil::parseString<double>(token);
                    stokenizer.nextToken();
                }
                rgbs.push_back(rgb);
            }

            vid++;
            continue;
        }

        if (token == "vt")
        {
            with_uv = true;
            CPoint2 uv;
            for (int i = 0; i < 2; i++)
            {
                stokenizer.nextToken();
                token = stokenizer.getToken();
                uv[i] = strutil::parseString<double>(token);
            }
            uvs.push_back(uv);
            continue;
        }

        if (token == "vn")
        {
            with_normal = true;

            CPoint n;
            for (int i = 0; i < 3; i++)
            {
                stokenizer.nextToken();
                token = stokenizer.getToken();
                n[i] = strutil::parseString<double>(token);
            }
            normals.push_back(n);
            continue;
        }

        if (token == "f")
        {
            sface face;
            std::vector<int> vert_ids(3);

            for (int i = 0; i < 3; i++)
            {
                svertex vertex;

                stokenizer.nextToken();
                token = stokenizer.getToken();

                strutil::Tokenizer tokenizer(token, " /\t\r\n");

                int ids[3];
                int k = 0;
                while (tokenizer.nextToken())
                {
                    std::string token = tokenizer.getToken();
                    ids[k] = strutil::parseString<int>(token);
                    k++;
                }

                vertex.vid = ids[0];
                if (with_uv)
                    vertex.tid = ids[1];
                if (with_normal)
                    vertex.nid = ids[2];
                face.verts.push_back(vertex);
                vert_ids[i] = vertex.vid;
            }

            sfaces.push_back(face);
            face_id_vids.insert(std::make_pair(fid++, vert_ids));
        }
    }
    fs.close();

    // 2. build mesh
    pMesh->load(vert_id_point, face_id_vids);

    // 3. set traits
    int vidx = 0;
    for (typename M::VertexIterator vit(pMesh); !vit.end(); ++vit)
    {
        typename M::CVertex* pV = *vit;
        vidx = pV->id() - 1;
        if(with_rgb)
            pV->rgb() = rgbs[vidx];
        if (with_normal)
            pV->normal() = normals[vidx];
    }

    if (with_uv)
    {
        for (typename M::FaceIterator fit(pMesh); !fit.end(); ++fit)
        {
            typename M::CFace* pF = *fit;
            int fid = pF->id();
            typename M::CDart* pD = pMesh->face_dart(pF);
            pD = pMesh->dart_prev(pD);

            for (int i = 0; i < 3; ++i)
            {
                int tid = sfaces[fid - 1].verts[i].tid;
                pD->uv() = uvs[tid - 1];
                pD->cid() = tid;
                pD = pMesh->dart_next(pD);
            }
        }
    }
}

template <class M>
void write(M* pMesh, const std::string& output)
{
    if (strutil::endsWith(output, ".m"))
        write_m<M>(pMesh, output);
    else if (strutil::endsWith(output, ".obj"))
        write_obj<M>(pMesh, output);
    else
    {
        std::cerr << "Not support to write to " << output << "\n";
        return;
    }
}

template <class M>
void write_m(M* pMesh, const std::string& output)
{
    for (typename M::VertexIterator viter(pMesh); !viter.end(); ++viter)
    {
        typename M::CVertex* pV = *viter;
        pV->to_string();
    }

    for (typename M::EdgeIterator eiter(pMesh); !eiter.end(); ++eiter)
    {
        typename M::CEdge* pE = *eiter;
        pE->to_string();
    }

    for (typename M::FaceIterator fiter(pMesh); !fiter.end(); ++fiter)
    {
        typename M::CFace* pF = *fiter;
        pF->to_string();
    }

    for (typename M::DartIterator diter(pMesh); !diter.end(); ++diter)
    {
        typename M::CDart* pD = *diter;
        pD->to_string();
    }

    // write to file
    std::fstream fs(output, std::fstream::out);
    if (fs.fail())
    {
        std::cerr << "Error in opening file " << output << "\n";
        return;
    }

    for (typename M::VertexIterator viter(pMesh); !viter.end(); ++viter)
    {
        typename M::CVertex* pV = *viter;
        fs << "Vertex " << pV->id();

        for (int i = 0; i < 3; i++)
        {
            fs << " " << pV->point()[i];
        }
        if (pV->string().size() > 0)
        {
            fs << " "
               << "{" << pV->string() << "}";
        }
        fs << std::endl;
    }

    for (typename M::FaceIterator fiter(pMesh); !fiter.end(); ++fiter)
    {
        typename M::CFace* pF = *fiter;
        fs << "Face " << pF->id();

        typename M::CDart* pD = pMesh->D(pF);
        do
        {
            fs << " " << pMesh->C0(pD)->id();
            pD = pMesh->beta(1, pD);
        } while (pD != pMesh->D(pF));

        if (!pF->string().empty())
        {
            fs << " {" << pF->string() << "}";
        }
        fs << std::endl;
    }

    for (typename M::EdgeIterator eiter(pMesh); !eiter.end(); ++eiter)
    {
        typename M::CEdge* pE = *eiter;
        if (!pE->string().empty())
        {
            fs << "Edge " << pMesh->edge_vertex(pE, 0)->id() << " " << pMesh->edge_vertex(pE, 1)->id() << " ";
            fs << "{" << pE->string() << "}" << std::endl;
        }
    }

    for (typename M::DartIterator diter(pMesh); !diter.end(); ++diter)
    {
        typename M::CDart* pD = *diter;
        if (!pD->string().empty())
        {
            fs << "Corner " << pMesh->C0(pD)->id() << " " << pMesh->C2(pD)->id() << " ";
            fs << "{" << pD->string() << "}" << std::endl;
        }
    }

    fs.close();
}

template<class M>
void write_obj(M* pMesh, const std::string& output)
{
    std::fstream fs(output, std::fstream::out);
    if (fs.fail())
    {
        fprintf(stderr, "Error is opening file %s\n", output.c_str());
        return;
    }

    int vid = 1;
    for (typename M::VertexIterator viter(pMesh); !viter.end(); ++viter)
    {
        typename M::CVertex* pV = *viter;
        pV->id() = vid++;
    }

    for (typename M::VertexIterator viter(pMesh); !viter.end(); ++viter)
    {
        typename M::CVertex* pV = *viter;

        fs << "v";
        for (int i = 0; i < 3; i++)
        {
            fs << " " << pV->point()[i];
        }
        fs << std::endl;
    }

    for (typename M::VertexIterator viter(pMesh); !viter.end(); ++viter)
    {
        typename M::CVertex* pV = *viter;

        fs << "vt";
        for (int i = 0; i < 2; i++)
        {
            fs << " " << pV->uv()[i];
        }
        fs << std::endl;
    }

    for (typename M::VertexIterator viter(pMesh); !viter.end(); ++viter)
    {
        typename M::CVertex* pV = *viter;

        fs << "vn";
        for (int i = 0; i < 3; i++)
        {
            fs << " " << pV->normal()[i];
        }
        fs << std::endl;
    }

    for (typename M::FaceIterator fiter(pMesh); !fiter.end(); ++fiter)
    {
        typename M::CFace* pF = *fiter;

        fs << "f";
        typename M::CDart* pD = pMesh->face_dart(pF);

        do {
            int vid = pMesh->dart_target(pD)->id();
            fs << " " << vid << "/" << vid << "/" << vid;
            pD = pMesh->dart_next(pD);
        } while (pD != pMesh->face_dart(pF));
        fs << std::endl;
    }

    fs.close();
}
} // namespace Dim2


/*===================================================================
                    IO functions for 3-manifold
===================================================================*/

namespace Dim3
{
template <class M>
void read(M* pMesh, const std::string& input);

template <class M>
void read_t(M* pMesh, const std::string& input);

template <class M>
void write(M* pMesh, const std::string& output);

template <class M>
void write_t(M* pMesh, const std::string& output);


template <class M>
void read(M* pMesh, const std::string& input)
{
    if (strutil::endsWith(input, ".t"))
        read_t<M>(pMesh, input);
    else
    {
        std::cerr << "Not support to read in " << input << "\n";
        return;
    }
}

template <class M>
void read_t(M* pMesh, const std::string& input)
{
    std::map<int, CPoint>        vert_id_point; // vid -> coordinate
    std::map<int, std::string>   vert_id_str;   // vid  -> string

    std::map<int, std::vector<int>> volume_id_vids; // volid -> vert_idx
    std::map<int, std::string>      volume_id_str;  // volid -> string

    std::vector<std::tuple<int, int, std::string>>   edge_attrs; //(vid1, vid2) -> string
    
    std::fstream is(input, std::fstream::in);
    if (is.fail())
    {
        std::cerr << "Error in opening file " << input << "\n";
        return;
    }

    char buffer[MAX_LINE];

    int nVertices = 0;
    int nVolumes = 0;
    int nEdges = 0;

    while (!is.eof())
    {
        is.getline(buffer, MAX_LINE);
        std::string line(buffer);
        line = strutil::trim(line);
        strutil::Tokenizer stokenizer(line, " \r\n");

        stokenizer.nextToken();
        std::string token = stokenizer.getToken();

        if (token == "Vertex")
            nVertices++;
        if (token == "Tet")
            nVolumes++;
        if (token == "Edge")
            nEdges++;
    }

    is.clear();                 // forget we hit the end of file
    is.seekg(0, std::ios::beg); // move to the start of the file

    int vid, volid;
    // read in vertices
    for (int i = 0; i < nVertices && is.getline(buffer, MAX_LINE); i++)
    {
        std::string line(buffer);
        line = strutil::trim(line);
        strutil::Tokenizer stokenizer(line, " \r\n");

        stokenizer.nextToken();
        std::string token = stokenizer.getToken();

        if (token != "Vertex")
        {
            i--;
            std::cerr << "Warning: File Format Error\n";
            continue;
        }

        stokenizer.nextToken();
        token = stokenizer.getToken();
        int vid = std::atoi(token.c_str()); // strutil::parseString<int>(token);

        CPoint p;
        for (int k = 0; k < 3; k++)
        {
            stokenizer.nextToken();
            std::string token = stokenizer.getToken();
            p[k] = std::atof(token.c_str()); // strutil::parseString<double>(token);
        }
        
        vert_id_point.insert(std::make_pair(vid, p));

        if (!stokenizer.nextToken("\t\r\n"))
            continue;
        token = stokenizer.getToken();

        int sp = (int) token.find("{");
        int ep = (int) token.find("}");

        if (sp >= 0 && ep >= 0)
        {
            std::string str = token.substr(sp + 1, ep - sp - 1);
            vert_id_str.insert(std::make_pair(vid, str));
        }
    }

    // read in tets
    for (int i = 0; i < nVolumes && is.getline(buffer, MAX_LINE); i++)
    {
        std::string line(buffer);
        line = strutil::trim(line);
        strutil::Tokenizer stokenizer(line, " \r\n");

        stokenizer.nextToken();
        std::string token = stokenizer.getToken();

        if (token != "Tet")
        {
            i--;
            std::cerr << "Warning: File Format Error\n";
            continue;
        }

        // skip the first "4" in the line
        stokenizer.nextToken();
        token = stokenizer.getToken();
        volid = std::atoi(token.c_str()); // strutil::parseString<int>(token);

        std::vector<int> vert_ids;

        for (int k = 0; k < 4; k++)
        {
            stokenizer.nextToken();
            std::string token = stokenizer.getToken();
            vid = std::atoi(token.c_str()); // strutil::parseString<int>(token);
            vert_ids.push_back(vid);
        }
        volume_id_vids.insert(std::make_pair(volid, vert_ids));

        // read in string
        if (!stokenizer.nextToken("\t\r\n"))
            continue;
        token = stokenizer.getToken();

        int sp = (int) token.find("{");
        int ep = (int) token.find("}");

        if (sp >= 0 && ep >= 0)
        {
            std::string str = token.substr(sp + 1, ep - sp - 1);
            volume_id_str.insert(std::make_pair(volid, str));
        }
    }

    // read in edge traits
    for (int i = 0; i < nEdges && is.getline(buffer, MAX_LINE); i++)
    {
        std::string line(buffer);
        line = strutil::trim(line);
        strutil::Tokenizer stokenizer(line, " \r\n");

        stokenizer.nextToken();
        std::string token = stokenizer.getToken();

        if (token != "Edge")
        {
            std::cerr << "Warning: File Format Error\n";
            return;
        }

        stokenizer.nextToken();
        token = stokenizer.getToken();
        int id0 = strutil::parseString<int>(token);

        stokenizer.nextToken();
        token = stokenizer.getToken();
        int id1 = strutil::parseString<int>(token);

        if (!stokenizer.nextToken("\t\r\n"))
        {
            continue;
        }

        token = stokenizer.getToken();

        int sp = (int) token.find("{");
        int ep = (int) token.find("}");

        std::string str;
        if (sp >= 0 && ep >= 0)
        {
            str = token.substr(sp + 1, ep - sp - 1);
        }
        edge_attrs.push_back(std::make_tuple(id0, id1, str));
    }

    is.close();

    pMesh->load(vert_id_point, volume_id_vids);

    pMesh->load_attributes(vert_id_str, volume_id_str, edge_attrs);
}

template <class M>
void write(M* pMesh, const std::string& output)
{
    if (strutil::endsWith(output, ".t"))
        write_t<M>(pMesh, output);
    else
    {
        std::cerr << "Not support to write in " << output << "\n";
        return;
    }
}

template <class M>
void write_t(M* pMesh, const std::string& output)
{
    for (typename M::VertexIterator viter(pMesh); !viter.end(); ++viter)
    {
        typename M::CVertex* pV = *viter;
        pV->to_string();
    }

    for (typename M::EdgeIterator eiter(pMesh); !eiter.end(); ++eiter)
    {
        typename M::CEdge* pE = *eiter;
        pE->to_string();
    }

    for (typename M::FaceIterator fiter(pMesh); !fiter.end(); ++fiter)
    {
        typename M::CFace* pF = *fiter;
        pF->to_string();
    }

    for (typename M::VolumeIterator voliter(pMesh); !voliter.end(); ++voliter)
    {
        typename M::CVolume* pVol = *voliter;
        pVol->to_string();
    }

    for (typename M::DartIterator diter(pMesh); !diter.end(); ++diter)
    {
        typename M::CDart* pD = *diter;
        pD->to_string();
    }

    // write to file
    std::fstream fs(output, std::fstream::out);
    if (fs.fail())
    {
        std::cerr << "Not support to write in " << output << "\n";
        return;
    }

    for (typename M::VertexIterator viter(pMesh); !viter.end(); ++viter)
    {
        typename M::CVertex* pV = *viter;
        fs << "Vertex " << pV->id();

        for (int i = 0; i < 3; i++)
        {
            fs << " " << pV->point()[i];
        }
        if (pV->string().size() > 0)
        {
            fs << " "
               << "{" << pV->string() << "}";
        }
        fs << std::endl;
    }

    for (typename M::VolumeIterator fiter(pMesh); !fiter.end(); ++fiter)
    {
        typename M::CVolume* pVol = *fiter;
        fs << "Tet " << pVol->id();

        typename M::CDart* pD = pMesh->D(pVol);
        do
        {
            fs << " " << pMesh->C0(pD)->id();
            pD = pMesh->beta(1, pD);
        } while (pD != pMesh->D(pVol));

        pD = pMesh->beta(1, pMesh->beta(2, pD));
        fs << " " << pMesh->C0(pD)->id();

        if (!pVol->string().empty())
        {
            fs << " {" << pVol->string() << "}";
        }
        fs << std::endl;
    }

    for (typename M::EdgeIterator eiter(pMesh); !eiter.end(); ++eiter)
    {
        typename M::CEdge* pE = *eiter;
        if (!pE->string().empty())
        {
            fs << "Edge " << pMesh->edge_vertex(pE, 0)->id() << " " << pMesh->edge_vertex(pE, 1)->id() << " ";
            fs << "{" << pE->string() << "}" << std::endl;
        }
    }

    fs.close();
}

} // namespace Dim3
} // namespace IO
} // namespace DartLib
#endif // !_DARTLIB_IO_H_
