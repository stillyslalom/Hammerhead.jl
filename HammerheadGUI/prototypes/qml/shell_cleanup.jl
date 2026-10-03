# Keep the original processing/render failure when disposal also fails.
module ShellCleanup
function report_failure(exception)
    println(stderr,"PROTOTYPE_CLEANUP_FAILURE")
    showerror(stderr,exception)
    println(stderr)
end
function with_cleanup(body,cleanup;report=report_failure)
    failed=false
    try
        body()
    catch
        failed=true
        rethrow()
    finally
        try
            cleanup()
        catch error
            failed || rethrow()
            captured=CapturedException(error,catch_backtrace())
            # Even diagnostic failure must not replace the original exception.
            try report(captured) catch diagnostic
                report_failure(CapturedException(diagnostic,catch_backtrace()))
            end
        end
    end
end
end
