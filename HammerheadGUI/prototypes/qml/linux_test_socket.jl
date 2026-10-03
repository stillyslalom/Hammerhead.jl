module LinuxTestSocket
function address(name)
    bytes=Vector{UInt8}(codeunits(name));length(bytes)<106 || error("abstract socket name too long")
    addr=zeros(UInt8,110)
    GC.@preserve addr unsafe_store!(Ptr{UInt16}(pointer(addr)),1)
    copyto!(addr,4,bytes,1,length(bytes))
    addr,3+length(bytes)
end
function socket()
    fd=ccall(:socket,Cint,(Cint,Cint,Cint),1,5|0x80000,0)
    fd>=0 || error("test socket errno=$(Libc.errno())")
    Int(fd)
end
function listen(name)
    fd=socket();addr,len=address(name)
    try
        ccall(:bind,Cint,(Cint,Ptr{UInt8},UInt32),fd,addr,len)==0 || error("test bind errno=$(Libc.errno())")
        ccall(:listen,Cint,(Cint,Cint),fd,1)==0 || error("test listen errno=$(Libc.errno())")
        flags=ccall(:fcntl,Cint,(Cint,Cint),fd,3)
        ccall(:fcntl,Cint,(Cint,Cint,Cint),fd,4,flags|0x800)==0 || error("test nonblocking listen")
        fd
    catch
        ccall(:close,Cint,(Cint,),fd);rethrow()
    end
end
function connect(name)
    fd=socket();addr,len=address(name)
    ccall(:connect,Cint,(Cint,Ptr{UInt8},UInt32),fd,addr,len)==0 || begin
        err=Libc.errno();ccall(:close,Cint,(Cint,),fd);error("test connect errno=$err")
    end
    fd
end
function accept(fd)
    peer=ccall(:accept4,Cint,(Cint,Ptr{Cvoid},Ptr{Cvoid},Cint),fd,C_NULL,C_NULL,0x80000|0x800)
    peer<0 && Libc.errno()==11 && return nothing
    peer>=0 || error("test accept errno=$(Libc.errno())")
    Int(peer)
end
end
