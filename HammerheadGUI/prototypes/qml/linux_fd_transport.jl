module LinuxFDTransport
using TOML
const MAX_PACKET=2048
struct IOVec
    base::Ptr{Cvoid}
    length::Csize_t
end
struct MsgHdr
    name::Ptr{Cvoid}
    namelen::UInt32
    iov::Ptr{IOVec}
    iovlen::Csize_t
    control::Ptr{Cvoid}
    controllen::Csize_t
    flags::Cint
end
function pair()
    Sys.islinux() && Sys.WORD_SIZE==64 || error("Linux fd transport requires 64-bit Linux")
    @assert sizeof(MsgHdr)==56 && fieldoffset(MsgHdr,3)==16 && sizeof(IOVec)==16
    fds=zeros(Cint,2)
    ccall(:socketpair,Cint,(Cint,Cint,Cint,Ptr{Cint}),1,5|0x80000,0,fds)==0 || error("socketpair errno=$(Libc.errno())")
    fds
end
function send_fd(socket,fd,data)
    io=IOBuffer();TOML.print(io,data;sorted=true);bytes=take!(io)
    0<length(bytes)<=MAX_PACKET || error("fd message exceeds bound")
    control=zeros(UInt8,24)
    GC.@preserve bytes control begin
        unsafe_store!(Ptr{Csize_t}(pointer(control)),20)
        unsafe_store!(Ptr{Cint}(pointer(control)+8),1)
        unsafe_store!(Ptr{Cint}(pointer(control)+12),1)
        unsafe_store!(Ptr{Cint}(pointer(control)+16),fd)
        iov=Ref(IOVec(pointer(bytes),length(bytes)))
        GC.@preserve iov begin
            msg=Ref(MsgHdr(C_NULL,0,Base.unsafe_convert(Ptr{IOVec},iov),1,pointer(control),length(control),0))
            rc=ccall(:sendmsg,Clong,(Cint,Ptr{MsgHdr},Cint),socket,msg,0x4000|0x40)
            rc==length(bytes) || error("sendmsg failed errno=$(Libc.errno())")
        end
    end
    nothing
end
function receive_fd(socket)
    bytes=zeros(UInt8,MAX_PACKET);control=zeros(UInt8,256);fds=Int[]
    try
        GC.@preserve bytes control begin
            iov=Ref(IOVec(pointer(bytes),length(bytes)))
            GC.@preserve iov begin
                msg=Ref(MsgHdr(C_NULL,0,Base.unsafe_convert(Ptr{IOVec},iov),1,pointer(control),length(control),0))
                rc=ccall(:recvmsg,Clong,(Cint,Ptr{MsgHdr},Cint),socket,msg,0x40|0x40000000)
                rc<0 && Libc.errno()==11 && return nothing
                rc>=0 || error("recvmsg failed errno=$(Libc.errno())")
                pos=0;last=Int(msg[].controllen)
                while pos+16<=last
                    len=Int(unsafe_load(Ptr{Csize_t}(pointer(control)+pos)))
                    16<=len<=last-pos || error("malformed fd ancillary header")
                    level=unsafe_load(Ptr{Cint}(pointer(control)+pos+8))
                    type=unsafe_load(Ptr{Cint}(pointer(control)+pos+12))
                    if level==1 && type==1
                        (len-16)%4==0 || error("malformed rights payload")
                        for offset in 16:4:len-4
                            push!(fds,Int(unsafe_load(Ptr{Cint}(pointer(control)+pos+offset))))
                        end
                    else
                        error("unexpected ancillary metadata")
                    end
                    pos+=(len+7)&~7
                end
                msg[].flags & (0x20|0x8)==0 || error("truncated fd message")
                rc==0 && isempty(fds) && return (;kind=:eof)
                rc>0 || error("zero-payload rights message")
                length(fds)==1 || error("exactly one pidfd required")
                data=TOML.parse(String(copy(bytes[1:rc])))
                fd=only(fds);empty!(fds)
                return (;kind=:fd,fd,data)
            end
        end
    finally
        for fd in fds
            ccall(:close,Cint,(Cint,),fd)
        end
    end
end
end
